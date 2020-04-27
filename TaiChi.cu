#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cfloat>
#include <errno.h>
#include "cusparse.h"
#include "mmio.h"
#include <dirent.h>
#include <sys/time.h>
#include "anonymouslib_cuda.h"
#include "TaiChi.h" 

using namespace std;

inline void checkcuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		printf("hello");
	}
}

inline void checkcusparse(cusparseStatus_t result)
{
	if(result != CUSPARSE_STATUS_SUCCESS){
		printf("CUSPARSE Error, error_code =  %d\n", result);
	}
}

int readMtx(char *filename, int &m, int &n, int &nnzA, int *&csrRowPtrA, int *&csrColIdxA,
	float *&csrValA)
{
	int ret_code = 0;
	MM_typecode matcode;

	FILE *f = NULL;
	int nnzA_mtx_report = 0;
	int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
	// load matrix
	if ((f = fopen(filename, "r")) == NULL)
		return -1;

	if (mm_read_banner(f, &matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		return -2;
	}

	if (mm_is_complex(matcode)) {
		printf("Sorry, data type 'COMPLEX' is not supported. \n");
		return -3;
	}

	if (mm_is_pattern(matcode)) {
		isPattern = 1; printf("type = Pattern.\n");
	}

	if (mm_is_real(matcode)) {
		isReal = 1; printf("type = real.\n");
	}

	if (mm_is_integer(matcode)) {
		isInteger = 1; printf("type = integer.\n");
	}

	ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
	if (ret_code != 0)
		return -4;

	if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
		isSymmetric = 1;
		printf("symmetric = true.\n");
	}
	else {
		printf("symmetric = false.\n");
	}

	int *csrRowPtrA_counter = (int *)malloc((m + 1) * sizeof(int));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

	int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
	memset(csrRowIdxA_tmp, 0, nnzA_mtx_report * sizeof(int));
	int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
	memset(csrColIdxA_tmp, 0, nnzA_mtx_report * sizeof(int));
	float *csrValA_tmp = (float *)malloc(nnzA_mtx_report * sizeof(float));
	memset(csrValA_tmp, 0.0, nnzA_mtx_report * sizeof(float));

	for (int i = 0; i < nnzA_mtx_report; i++)
	{
		int idxi = 0, idxj = 0;
		double fval = 0.0;
		int ival = 0;

		if (isReal)
			fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
		else if (isInteger)
		{
			fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
			fval = ival;
		}
		else if (isPattern)
		{
			fscanf(f, "%d %d\n", &idxi, &idxj);
			fval = 1.0;
		}

		// adjust from 1-based to 0-based
		idxi--;
		idxj--;

		csrRowPtrA_counter[idxi]++;
		csrRowIdxA_tmp[i] = idxi;
		csrColIdxA_tmp[i] = idxj;
		csrValA_tmp[i] = fval;
	}

	if (f != stdin)
		fclose(f);	

	if (isSymmetric)
	{
		for (int i = 0; i < nnzA_mtx_report; i++) {
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
		}
	}

	// exclusive scan for csrRowPtrA_counter
	int old_val = 0, new_val = 0;

	old_val = csrRowPtrA_counter[0];
	csrRowPtrA_counter[0] = 0;
	for (int i = 1; i <= m; i++)
	{
		new_val = csrRowPtrA_counter[i];
		csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
		old_val = new_val;
	}

	nnzA = csrRowPtrA_counter[m];
	csrRowPtrA = (int *)malloc((m + 1) * sizeof(int));
	memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

	csrColIdxA = (int *)malloc(nnzA * sizeof(int));
	memset(csrColIdxA, 0, nnzA * sizeof(int));
	csrValA = (float *)malloc(nnzA * sizeof(float));
	memset(csrValA, 0, nnzA * sizeof(float));

	if (isSymmetric)
	{
		for (int i = 0; i < nnzA_mtx_report; i++)
		{
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
			{
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

				offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
				csrColIdxA[offset] = csrRowIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
			}
			else
			{
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
			}
		}
	}
	else
	{
		for (int i = 0; i < nnzA_mtx_report; i++)
		{
			int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
			csrColIdxA[offset] = csrColIdxA_tmp[i];
			csrValA[offset] = csrValA_tmp[i];
			csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
		}
	}

	// free tmp space
	free(csrColIdxA_tmp);
	free(csrValA_tmp);
	free(csrRowIdxA_tmp);
	free(csrRowPtrA_counter);
	return 0;
}

void cal_shape_size(int M, int N, int mF, int x1, int y1, int a1, int b1, int &x, int &y, int &a, int &b, float &gain)
{
	if (M > mF) {
		gain = (float)M/(float)(mF);
		x = (int)(x1 * gain);
		y = (int)(y1 * gain);
		a = (int)(a1 * gain);
		b = (int)(b1 * gain);
	}
	else if (M < mF) {
		gain = (float)mF/(float)M;
		x = (int)(x1 / gain);
		y = (int)(y1 / gain); 
		a = (int)(a1 / gain);
		b = (int)(b1 / gain);
	}
	else { // M == mF
		gain = 1.0;   // 如果直接写成(float)M/(float)mF,可能会得到结果为inf
		x = x1;
		y = y1; 
		a = a1;
		b = b1;
	}
	if (x < 0) x = 0; if (x > M-1) x = M-1; 
	if (y < 0) y = 0; if (y > N-1) y = N-1;
	if (y + a > N) a = N - y; 
	if (x + b > M) b = M - x;
}

void partition_shapes(FILE *fSdf, int M, int N, int nnz, int* &csrRowIndexHostPtr, int* &csrColIndexHostPtr, float* &csrValHostPtr, int mF, int nF, int num_shapes, shape *h, TaiChi *neg_format)
{
	for(int i = 0; i<num_shapes + 1; i++){
		printf("\nBegin %2d-th shape: ", i);
		h[i].id = i;
		if(fscanf(fSdf, "%s %d %d %d %d %d", &(h[i].category), &(h[i].x1), &(h[i].y1), &(h[i].a1), &(h[i].b1), &(h[i].area1)) != EOF)
		{
			h[i].x1--;	//x1是以one-based的，而x是以zero-based的
			h[i].y1--;
			float gain;

			cal_shape_size(M, N, mF, h[i].x1, h[i].y1, h[i].a1, h[i].b1, h[i].x, h[i].y, h[i].a, h[i].b, gain);
			if(strcmp(h[i].category, "diagonal") == 0) {
				printf("diagonal.\n");
				partition_dia_shape(M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr, gain, h[i], neg_format[i]);
			} // end diagonal
			else {
				//矩形和三角形
				printf("rectangle/triangle\n");
				neg_format[i].neg = 0;
				neg_format[i].maxNumZero = 0;
				neg_format[i].shape = 2; // 2 means rectangle or triangle
				neg_format[i].nnz = 0;
				neg_format[i].width = 0;
				neg_format[i].M_sub = M;
				neg_format[i].N_sub = N;
				neg_format[i].xStart = 0;
				neg_format[i].yStart = 0;

				h[i].nnz_submatrix = 0;
				h[i].b = 0;
				h[i].a = 0;
				h[i].x = 0;
				h[i].y = 0;
			}// end rectangle and triangle
		} 
		else { //ultra-sparse shape
			printf("ultra-sparse shape\n");

			int sparse_nnz = nnz;
			for (int j = 0; j < num_shapes; j++) sparse_nnz -= neg_format[j].nnz;
			printf("sparse_nnz = %d\n", sparse_nnz);

			if (sparse_nnz < nnz) {
				neg_format[i].csrRow = (int *)malloc((M+1) * sizeof(int));
				neg_format[i].csrCol = (int *)malloc(sparse_nnz * sizeof(int));
				// memset(neg_format[i].csrRow, 0, (M+1) * sizeof(int));
				// memset(neg_format[i].csrCol, 0, sparse_nnz * sizeof(int));
                for (int j = 0; j < M+1; j++) neg_format[i].csrRow[j] = 0;
                for (int j = 0; j < sparse_nnz; j++) neg_format[i].csrCol[j] = 0;
				
				int n  = 0;
				// 摘取剩余没有被标记的非零元
				for (int r = 0; r < M; r++) {
					for (int j = csrRowIndexHostPtr[r]; j < csrRowIndexHostPtr[r+1]; j++) {
						if ( fabs(csrValHostPtr[j]-FLT_MAX) > ZERO ) {
							neg_format[i].csrRow[r] ++;
							neg_format[i].csrCol[n++] = csrColIndexHostPtr[j];
						}
					}
				}
				if (n != sparse_nnz) {printf("PARTITION WRONG!!");}
                int old_val, new_val;
                old_val = neg_format[i].csrRow[0];
                neg_format[i].csrRow[0] = 0;
                for (int j = 1; j < M+1; j++)
                {
                    new_val = neg_format[i].csrRow[j];
                    neg_format[i].csrRow[j] = old_val + neg_format[i].csrRow[j-1];
                    old_val = new_val;
                }
			}
			else {
				neg_format[i].csrRow = csrRowIndexHostPtr;
				neg_format[i].csrCol = csrColIndexHostPtr;
			}

            // printf("csrRow:");
            // for (int j = 0; j <= M; j++) printf("csrRow[%d]:%d\t", j, neg_format[i].csrRow[j]);
            // printf("\ncsrCol:");
            // for (int j = 0; j < sparse_nnz; j++) printf("%d\t", neg_format[i].csrCol[j]);
            // printf("\n");
			h[i].nnz_submatrix = sparse_nnz;
			h[i].b = M;
			h[i].a = N;
			h[i].x = 0;
			h[i].y = 0;

			neg_format[i].shape = 1; // 1 means ultra-sparse shape
			neg_format[i].neg = 0;
			neg_format[i].maxNumZero = 0;
			neg_format[i].M_sub = M;
			neg_format[i].N_sub = N;
			neg_format[i].xStart = 0;
			neg_format[i].yStart = 0;
			neg_format[i].width = 0;
			neg_format[i].nnz = sparse_nnz;
			printf("nnz = %d\n", sparse_nnz);
		} //end else
		printf("End %2d-th shape.\n", i);
	}
}

void partition_dia_shape(int M, int N, int nnz, int* &csrRowIndexHostPtr, int* &csrColIndexHostPtr, float* &csrValHostPtr, float gain, shape &hi, TaiChi &neg_format_i)
{
	neg_format_i.M_sub = hi.b;
	neg_format_i.N_sub = hi.a;
	neg_format_i.xStart = hi.x;
	neg_format_i.yStart = hi.y;
	int xStart = neg_format_i.xStart, yStart = neg_format_i.yStart;
	int M_sub = neg_format_i.M_sub, N_sub = neg_format_i.N_sub;

	int width;
	// 粗略估计对角块的宽度
	int smaller_dim = (M_sub > N_sub)? N_sub: M_sub;
	if (smaller_dim>= 10000000)
		width = smaller_dim * 0.000001;
	else if (smaller_dim >= 1000000)
		width = smaller_dim * 0.00001;
	else if (smaller_dim >= 100000)
		width = smaller_dim * 0.001;
	else if (smaller_dim >= 10000)
		width = smaller_dim * 0.01;
	else 
		width = smaller_dim * 0.1;

	/* 按照公式求出来的width太小
	int width = h[i].a - sqrt(h[i].a*h[i].a-h[i].area1*gain) + 2;
	int width = h[i].a - sqrt(h[i].a*h[i].a-h[i].area1*gain)+20 ;
	*/

	printf("gain=%.6f, width=%d\n", gain, width);
	printf("x1:%d,y1:%d, a1:%d, b1:%d\n", hi.x1, hi.y1, hi.a1, hi.b1);
	printf("x:%d, y:%d, a:%d, b:%d\n", hi.x, hi.y, hi.a, hi.b);
	
	neg_format_i.width = width;
	neg_format_i.shape = 0; // 0 means diagonal
	int num_diagonals = 2*width-1;
	
	int ** rowID_nonZero_each_dia = NULL; //存储每一个对角线中非零元的行索引
	rowID_nonZero_each_dia = (int **) malloc(sizeof(int*)*num_diagonals);
	memset(rowID_nonZero_each_dia, 0, sizeof(int*)*num_diagonals);
	for (int k = 0; k < num_diagonals; k++)
	{
		rowID_nonZero_each_dia[k] = NULL;
		rowID_nonZero_each_dia[k] = (int *) malloc(sizeof(int) * smaller_dim);
		memset(rowID_nonZero_each_dia[k], 0, sizeof(int) * smaller_dim);
	}

	int *nnz_each_dia = NULL;
	nnz_each_dia = (int*)malloc(sizeof(int)*num_diagonals);
	memset(nnz_each_dia, 0, sizeof(int)*num_diagonals);

	int n_diagonal = 0;
	int diagonal_offset = hi.y - hi.x;
	int r = 0, c = 0, nnz_shape = 0;
	// 扫描非零元得到对角块 改用CSR格式
	for (int i = 0; i < M; i++) {
		for (int j = csrRowIndexHostPtr[i]; j < csrRowIndexHostPtr[i+1]; j++) {
			r = i; c = csrColIndexHostPtr[j];
			n_diagonal = c - r - (yStart - xStart) + width - 1;
			int new_r = r + diagonal_offset;
			if(	r >= hi.x && r < (hi.x+hi.b) && c >= hi.y && c < (hi.y+hi.a)
				&& c >= (new_r-width+1) && c <= (new_r+width-1) 
				&& fabs(csrValHostPtr[j]-FLT_MAX)>ZERO ){
     //           printf("r=%d,c=%d\n", r, c);
				// 该非零元在对角块内
				rowID_nonZero_each_dia[n_diagonal][nnz_each_dia[n_diagonal]++] = r - hi.x;
			}	
		}// end for j
	} // end for i

	int *neg_offsets = NULL;
	// save the offset of each dense diagonal
	neg_offsets = (int*)malloc(sizeof(int) * num_diagonals);
	memset(neg_offsets, 0, sizeof(int) * num_diagonals);
	int *start  = NULL, *end = NULL; // save the start and end row index of each dense diagonal
	start = (int*)malloc(sizeof(int) * num_diagonals);
	end   = (int*)malloc(sizeof(int) * num_diagonals);
	memset(start, 0, sizeof(int) * num_diagonals);
	memset(end,   0, sizeof(int) * num_diagonals);

	int neg = 0;
	int dia_offsets_temp = 0;
	float density_temp = 0.0;
	int start_temp = 0, end_temp = 0;
	
	// Find dense diagonals;
	for(int n = 0; n < num_diagonals; n++)
	{
		dia_offsets_temp = n - width + 1;
		cal_start_end(M_sub, N_sub, width, dia_offsets_temp, n, start_temp, end_temp);
		density_temp = float(nnz_each_dia[n]) / float(end_temp - start_temp);
		if (density_temp >= 0.6)
		{
			start[neg] = start_temp;
			end[neg] = end_temp;
			neg_offsets[neg] = dia_offsets_temp;
			neg ++;
		}
	}

    // for (int i = 0; i < neg; i++) printf("neg_offset[%d]=%d.\n", i, neg_offsets[i]);
	neg_format_i.neg = neg;
	//处理非零元密度大于阈值的对角线
	if (neg > 0) {
		neg_format_i.neg_offsets = (int *)malloc(sizeof(int)*neg);
		memcpy(neg_format_i.neg_offsets, neg_offsets, neg * sizeof(int));
		neg_format_i.start = (int*)malloc(sizeof(int)*neg);
		memcpy(neg_format_i.start, start, neg * sizeof(int));
		neg_format_i.end = (int*)malloc(sizeof(int)*neg);
		memcpy(neg_format_i.end, end, neg * sizeof(int));

		int maxNumZero=0; //存储单条稠密对角线中最大零元数
		//存储每一条稠密对角线的零元个数
		neg_format_i.numZeroNeg = (int *)malloc(sizeof(int)*neg);
		neg_format_i.cStart = (int *)malloc(sizeof(int)*neg);

		// 获取稠密对角线的最大零元数量和每一条稠密对角线的零元数量
		for(int d = 0; d < neg; d++)
		{
			n_diagonal = neg_format_i.neg_offsets[d] + width - 1;
			nnz_shape += nnz_each_dia[n_diagonal];
			neg_format_i.numZeroNeg[d] = neg_format_i.end[d] - neg_format_i.start[d] - nnz_each_dia[n_diagonal];
			maxNumZero = (neg_format_i.numZeroNeg[d] > maxNumZero)? neg_format_i.numZeroNeg[d]:maxNumZero;
			if(neg_format_i.neg_offsets[d] > 0)
				neg_format_i.cStart[d] = neg_format_i.neg_offsets[d];
			else
				neg_format_i.cStart[d] = 0;
			
		}
		neg_format_i.nnz = nnz_shape; 
		neg_format_i.maxNumZero = maxNumZero;

		// rowZeroNeg 存放每条稠密对角线中每个零元的行索引
		neg_format_i.rowZeroNeg = (int **)malloc(sizeof(int*)*neg);
		memset(neg_format_i.rowZeroNeg, 0, sizeof(int*)*neg);
		if(maxNumZero > 0)
		{
			for (int k = 0; k < neg; k++)
			{
				neg_format_i.rowZeroNeg[k] = NULL;
				neg_format_i.rowZeroNeg[k] = (int *) malloc(sizeof(int) * maxNumZero);
			}
		}

		//获取每条稠密对角线中每个零元的行索引
		for(int d = 0; d < neg; d++)
		{
			int rNnzIndex = 0;
			int n_diagonal = neg_offsets[d] + width -1;
			int rZeroIndex = 0;
			for (int r = neg_format_i.start[d]; r < neg_format_i.end[d]; r++)
			{
				if (r < rowID_nonZero_each_dia[n_diagonal][rNnzIndex])
				{
					neg_format_i.rowZeroNeg[d][rZeroIndex++] = r;
				}
				else
				{
					if(rNnzIndex < nnz_each_dia[n_diagonal])
						rNnzIndex ++;
					else
						neg_format_i.rowZeroNeg[d][rZeroIndex++] = r;
				}
					
			}
		}
	} // end if neg>0

	for (int k = 0; k < num_diagonals; k++)
	{
		if(rowID_nonZero_each_dia[k] != NULL)
		{
			free(rowID_nonZero_each_dia[k]);
			rowID_nonZero_each_dia[k] = NULL;
		}
	}
	if(rowID_nonZero_each_dia != NULL)
	{
		free(rowID_nonZero_each_dia);
		rowID_nonZero_each_dia = NULL;
	}
	if(nnz_each_dia != NULL) {free(nnz_each_dia); nnz_each_dia = NULL;}
	if(start != NULL) {free(start); start = NULL;}
	if(end != NULL) {free(end); end = NULL;}

    printf("neg:%d maxNumZero:%d nnz_shape:%d\n", neg, neg_format_i.maxNumZero, nnz_shape);
    // printf("start:");
    // for (int i=0; i < neg; i++) printf("%d ", neg_format_i.start[i]);
    // printf("\nend:");
    // for (int i=0; i < neg; i++) printf("%d ", neg_format_i.end[i]);
    // printf("\ncStart:");
    // for (int i=0; i < neg; i++) printf("%d ", neg_format_i.cStart[i]);
    // printf("\nneg_offsets:");
    // for (int i=0; i < neg; i++) printf("%d ", neg_format_i.neg_offsets[i]);
    // printf("\nnumZeroNeg:");
    // for (int i=0; i < neg; i++) printf("%d ", neg_format_i.numZeroNeg[i]);
    // printf("\nrowZeroNeg:");
    // for (int i=0; i < neg; i++)  {
    //     for (int j = 0; j < neg_format_i.numZeroNeg[i]; j++)
    //         printf("%d ", neg_format_i.rowZeroNeg[i][j]);
    //     printf("\n");
    // }
	// 标记包含在稠密对角块中的非零元
	if (neg > 0)
	{
		nnz_shape = 0;
		for (int i = 0; i < M; i++) {
			for (int j = csrRowIndexHostPtr[i]; j < csrRowIndexHostPtr[i+1]; j++) {
				r = i;
				c = csrColIndexHostPtr[j];
				int new_r = r + diagonal_offset;
				if(	r >= hi.x && r < (hi.x+hi.b) && c >= hi.y && c < (hi.y+hi.a)
					&& c >= (new_r-width+1) && c <= (new_r+width-1) 
					&& fabs(csrValHostPtr[j]-FLT_MAX)>ZERO ) {
					int r_sub = r - hi.x;
					int c_sub = c - hi.y;
					for(int d = 0; d <neg; d++)
					{	
						if(c_sub - r_sub == neg_offsets[d])
						{					
							nnz_shape++;
							csrValHostPtr[j] = FLT_MAX;//把采摘完的非零元做个标记
						}
					}
				}
			}
		}
        if (nnz_shape != neg_format_i.nnz) printf("This shape's PARTITION IS WRONG!!\n");
	}
	if(neg_offsets != NULL) {free(neg_offsets); neg_offsets = NULL;}
}

void cal_start_end (int M_sub, int N_sub, int width, int dia_offsets_temp, int n, int &start_temp, int &end_temp)
{
	if(M_sub >= N_sub) {
		if(n >= width -1) {
			start_temp = 0;
			end_temp = N_sub - dia_offsets_temp;
		}
		else {
			if(abs(dia_offsets_temp) > (M_sub - N_sub)) {
				start_temp = abs(dia_offsets_temp);
				end_temp = M_sub;
			}
			else {
				start_temp = abs(dia_offsets_temp);
				end_temp = N_sub + abs(dia_offsets_temp);
			}
		}
	}
	else if(M_sub < N_sub) {
		if(n <= width - 1) {
			start_temp = abs(dia_offsets_temp);
			end_temp = M_sub;
		}
		else {
			if(abs(dia_offsets_temp) > (N_sub - M_sub)) {
				start_temp = 0;
				end_temp = N_sub - abs(dia_offsets_temp);
			}
			else {
				start_temp = 0;
				end_temp = M_sub;
			}
		}
	}
}


int cal_memory(int num_shapes, TaiChi *neg_format)
{
	int memory_size = 0;
	memory_size += sizeof(TaiChi) * num_shapes;
	for(int i = 0; i < num_shapes + 1; i++)
	{
		if(neg_format[i].shape == 0)	// diagonal shape
		{
			memory_size += neg_format[i].neg*5 + neg_format[i].neg*neg_format[i].maxNumZero;
		}
		else if(neg_format[i].shape == 1)	// ultra-sparse shape
		{
			memory_size += neg_format[i].M_sub + 1 + neg_format[i].nnz;
		}
	}
	return memory_size;
}

void queryDevice()  
{  
    cudaDeviceProp deviceProp;  
    int deviceCount = 0;  
    cudaError_t cudaError;  
    cudaError = cudaGetDeviceCount(&deviceCount);  
	cout<<"cudaError = "<<cudaError<<endl;
    for (int i = 0; i < deviceCount; i++)  
    {  
        cudaError = cudaGetDeviceProperties(&deviceProp, i);  
        cout << "Device " << i << "'s main property: " << endl;  
        cout << "Device Name: " << deviceProp.name << endl;  
        cout << "Global Memory of device: " << deviceProp.totalGlobalMem / 1024 / 1024 << " MB" << endl;  
        cout << "Maximal available shared memory for a block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << endl;  
        cout << "Number of available registers for a block: " << deviceProp.regsPerBlock << endl;  
        cout << "Maximal number of threads for a block: " << deviceProp.maxThreadsPerBlock << endl;  
        cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << endl;  
        cout << "Number of multi processors: " << deviceProp.multiProcessorCount << endl;  
    }  
	cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)  printf("cudaSetDevice failed!");

	int device = -1;
	cudaStatus = cudaGetDevice(&device);
    if (cudaStatus != cudaSuccess)  printf("cudaGetDevice failed!");
	cout<<"\nThe device now beening used is device "<<device<<endl<<endl;
}   

int max(int *a, int len)
{
	int max = a[0];
	for(int i = 1; i < len; i ++)
		if(a[i] > max)
			max = a[i];
	return max;
}

int min(int *a, int len)
{
	int min = a[0];
	for(int i = 1; i < len; i ++)
		if(a[i] < min)
			min = a[i];
	return min;
}

// shape dia 
// fisrt set (all element involved in Y) to (X)
__global__ static void SpMV_dia_add(int m,
									int size_neg,
									int dia_shapes, 
									int *xStart_device,
									int *yStart_device,
									int *neg_device,
									int *end_add_tasks_device,
									int *start_device, 
									int *end_device, 
									int *cstart_device, 
									float *x_device, 
									float *y_sub_device)
{
	int my_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int sum_neg = 0;

	for(int j = 0; j < dia_shapes; j++)
	{
		if ((j == 0 && my_id < end_add_tasks_device[j]) || (j != 0 && my_id >= end_add_tasks_device[j-1] && my_id < end_add_tasks_device[j]))
		{
			int neg_format_index = j;
			int neg = neg_device[j];
			int start_tasks = ((j==0)? 0: end_add_tasks_device[j-1]);
			// length of longest diagonal line
			int maxNumOne = (end_add_tasks_device[j] - start_tasks) / neg; 
			//local index of diagonals
			int local_neg_index = (my_id - start_tasks) / maxNumOne;	
			// global index of diagonals, for indexing start,end, and neg_offsets arrays
			int global_neg_index = sum_neg + local_neg_index;	
			int end = end_device[global_neg_index];
			int start = start_device[global_neg_index];
			int xStart = xStart_device[j];
			int yStart = yStart_device[j];

			if (my_id - local_neg_index * maxNumOne -  start_tasks < end - start)
			{
				int y_index = global_neg_index * m + xStart + (my_id - local_neg_index * maxNumOne - start_tasks) + start;
				int x_index = yStart + cstart_device[global_neg_index] + (my_id - local_neg_index * maxNumOne - start_tasks);
				y_sub_device[y_index] += x_device[x_index];
			}
		}
		sum_neg += neg_device[j];
	}
}


// __device__ volatile int g_mutex = 0;
// shape dia
__global__ static void SpMV_dia_sub(int m, int size_neg,
									int dia_shapes,
									int *xStart_device,
									int *yStart_device,
									int *neg_device,
									int *maxNumZero_device,
									int *end_sub_tasks_device,
									int *neg_offsets_device, 
									int *numZeroNeg_device, 
									int *rowZeroNeg_device, 
									float *x_device, 
									float *y_sub_device)
{
	// here we have nnz works to be done
	int my_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	int sum_neg = 0;

	for(int j = 0; j < dia_shapes; j++)
	{
		if ((j == 0 && my_id < end_sub_tasks_device[j]) || (j != 0 && my_id >= end_sub_tasks_device[j-1] && my_id < end_sub_tasks_device[j]))
		{
			int start_tasks = ((j==0)? 0: end_sub_tasks_device[j-1]);
			int maxNumZero = maxNumZero_device[j];
			int local_neg_index = (my_id - start_tasks) / maxNumZero;	// local index of diagonals
			int global_neg_index = sum_neg + local_neg_index;	// global index of diagonals, indexing for end,start,neg_offsets arrays
			int numZero = numZeroNeg_device[global_neg_index];
			int xStart = xStart_device[j];
			int yStart = yStart_device[j];

			if (my_id - local_neg_index * maxNumZero -start_tasks < numZero)
			{
				int y_index = global_neg_index * m + xStart + rowZeroNeg_device[my_id];
				int x_index = yStart + rowZeroNeg_device[my_id] + neg_offsets_device[global_neg_index];
				y_sub_device[y_index] -= x_device[x_index];
			}
		}
		sum_neg += neg_device[j];
	}
}

// gather the Y vector from given pointer
__global__ static void SpMV_gather(float *Y, int Y_num, int Y_size, float *Y_sum)
{
	// here we have nnz works to be done
	int my_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(my_id < Y_size)
	{
		for(int i = 0; i < Y_num; i++){
			Y_sum[my_id] += Y[i * Y_size + my_id];
		}
	}
}

int call_anonymouslib(int m, int n, int nnzA,
	int *csrRowPtrA, int *csrColIdxA,
	VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
	int err = 0;
	cudaError_t err_cuda = cudaSuccess;

	double gb = getB<int, VALUE_TYPE>(m, nnzA);
	double gflop = getFLOP<int>(nnzA);

	// Define pointers of matrix A, vector x and y
	int *d_csrRowPtrA;
	int *d_csrColIdxA;
	// VALUE_TYPE *d_csrValA;
	VALUE_TYPE *d_x;
	VALUE_TYPE *d_y;
 
	// Matrix A
	checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m+1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnzA  * sizeof(int)));
	// checkCudaErrors(cudaMalloc((void **)&d_csrValA,    nnzA  * sizeof(VALUE_TYPE)));

    anonymouslib_timer transfer_timer;
    transfer_timer.start();
    for (int i = 0; i < NUM_RUN; i++)
    {
	    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int),   cudaMemcpyHostToDevice));
	    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnzA  * sizeof(int),   cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(d_csrValA,    csrValA,    nnzA  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice));
    }
	double transfer_time = transfer_timer.stop() / NUM_RUN;

	// Vector x
	checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(VALUE_TYPE)));
	checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

	// Vector y
	checkCudaErrors(cudaMalloc((void **)&d_y, m  * sizeof(VALUE_TYPE)));
	checkCudaErrors(cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE)));

	anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
	err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA);
	//err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA);
	//cout << "inputCSR err = " << err << endl;

	err = A.setX(d_x); // you only need to do it once!
	//cout << "setX err = " << err << endl;
 
	A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

	// warmup device
	// A.warmup();

	anonymouslib_timer asCSR5_timer;
	asCSR5_timer.start();

	err = A.asCSR5();

	cout << "CSR->CSR5 time = " << asCSR5_timer.stop() << " ms." << endl;
	//cout << "asCSR5 err = " << err << endl;

	// check correctness by running 1 time
	err = A.spmv(alpha, d_y);
	//cout << "spmv err = " << err << endl;
	checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

	// warm up by running 50 times
	// if (NUM_RUN)
	// {
	// 	for (int i = 0; i < 50; i++)
	// 	err = A.spmv(alpha, d_y);
	// }

	err_cuda = cudaDeviceSynchronize();

	anonymouslib_timer CSR5Spmv_timer;
	CSR5Spmv_timer.start();

	// time spmv by running NUM_RUN times
	for (int i = 0; i < NUM_RUN; i++) {
        checkCudaErrors(cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE)));
		err = A.spmv(alpha, d_y);
    }
	err_cuda = cudaDeviceSynchronize();

	double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;

	checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

	if (NUM_RUN)
	cout << "CSR5-based SpMV time = " << CSR5Spmv_time
	<< " ms. Bandwidth = " << gb/(1.0e+6 * CSR5Spmv_time)
	<< " GB/s. GFlops = " << gflop/(1.0e+6 * CSR5Spmv_time)  << " GFlops." << endl;

	// FILE  *fresult = fopen("TaiChi_CSR5_cuda_time.txt", "a+");
    // if (fresult == NULL){printf("Create file failed.\n ");}
    // fprintf(fresult, "CSR5 transmission time %.6f CSR5-based SpMV time %.6f ", transfer_time, CSR5Spmv_time);
	// fclose(fresult);
	A.destroy();

	checkCudaErrors(cudaFree(d_csrRowPtrA));
	checkCudaErrors(cudaFree(d_csrColIdxA));
	// checkCudaErrors(cudaFree(d_csrValA));
	checkCudaErrors(cudaFree(d_x));

	return err;
}

void spmv_stream_gpu(int M, int N, TaiChi *neg_format, int num_shapes, float *xHostPtr, float *yHostPtr)
{
	int size_neg = 0;	// save the total number of dense diagonals
	int size_negXmaxNumZero = 0;	// save the sum of neg * maxNumZero of all shapes
	int shape_index_len = 0;	// save the number of shapes including dense diagonals
	int shape_index[num_shapes+1];

	int dia_shapes = 0;	// save the number of diagonal shapes
	for (int i = 0; i < num_shapes + 1; i++)
	{
		if(neg_format[i].shape == 0 && neg_format[i].neg > 0)
		{
			// diagonal
			size_neg += neg_format[i].neg;
			shape_index[shape_index_len ++] = i;
			size_negXmaxNumZero += neg_format[i].neg * neg_format[i].maxNumZero;
			dia_shapes ++;
		}
		if(neg_format[i].shape == 1 && neg_format[i].nnz > 0)
		{
			// ultra sparse
			shape_index[shape_index_len ++] = i;
		}
	}

	float time_elapsed = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    printf("size_neg:%d, dia_shapes:%d, shape_index_len:%d, size_negXmaxNumZero:%d\n", size_neg, dia_shapes, shape_index_len, size_negXmaxNumZero);
	
	// now we have the total number of valid shapes -- shape_index_len 
	// and their index -- shape_index
	
	// move data for copying data from CPU to GPU
	int total_sub_tasks = 0;				// total tasks of subtraction of all diagonal blocks 
	int total_add_tasks = 0;				// total tasks of addition of all diagonal blocks
	int end_add_tasks[dia_shapes], end_sub_tasks[dia_shapes];		// tasks of addition of each diagonal block
	int *end_add_tasks_device = 0, *end_sub_tasks_device = 0;		// tasks of subtraction of each diagonal block
	
	int *neg_offsets_host = 0, *numZeroNeg_host = 0, *start_host = 0, *end_host = 0, *cstart_host = 0, *rowZeroNeg_host = 0;
	int *neg_offsets_device = 0, *numZeroNeg_device = 0, *start_device = 0, *end_device = 0, *cstart_device = 0, *rowZeroNeg_device = 0;
	
	int xStart_host[dia_shapes], yStart_host[dia_shapes], neg_host[dia_shapes], maxNumZero_host[dia_shapes];
	int *xStart_device = 0, *yStart_device = 0, *neg_device = 0, *maxNumZero_device = 0;

	// allocate space
	if (dia_shapes > 0) // exist dense diagonal shapes
	{
		// malloc space on CPU for add operation 
		checkcuda(cudaMallocHost((void**)&start_host, size_neg * sizeof(int)));
		checkcuda(cudaMallocHost((void**)&end_host, size_neg * sizeof(int)));
		checkcuda(cudaMallocHost((void**)&cstart_host, size_neg * sizeof(int)));

		// malloc space for add operation on GPU
		checkcuda(cudaMalloc((void**)&end_add_tasks_device, (dia_shapes) * sizeof(int)));
		checkcuda(cudaMalloc((void**)&start_device, size_neg * sizeof(int)));
		checkcuda(cudaMalloc((void**)&end_device, size_neg * sizeof(int)));
		checkcuda(cudaMalloc((void**)&cstart_device, size_neg * sizeof(int)));
		checkcuda(cudaMalloc((void**)&xStart_device, (dia_shapes) * sizeof(int)));
		checkcuda(cudaMalloc((void**)&yStart_device, (dia_shapes) * sizeof(int)));
		checkcuda(cudaMalloc((void**)&neg_device, (dia_shapes) * sizeof(int)));

		if (size_negXmaxNumZero > 0)
		{
			// malloc space for add operation on CPU
			checkcuda(cudaMallocHost((void**)&neg_offsets_host, size_neg * sizeof(int)));
			checkcuda(cudaMallocHost((void**)&numZeroNeg_host, size_neg * sizeof(int)));
			checkcuda(cudaMallocHost((void**)&rowZeroNeg_host, size_negXmaxNumZero * sizeof(int)));

			// malloc space for sub operation on GPU
			checkcuda(cudaMalloc((void**)&neg_offsets_device, size_neg * sizeof(int)));
			checkcuda(cudaMalloc((void**)&numZeroNeg_device, size_neg * sizeof(int)));
			checkcuda(cudaMalloc((void**)&rowZeroNeg_device, size_negXmaxNumZero * sizeof(int)));
			checkcuda(cudaMalloc((void**)&end_sub_tasks_device, (dia_shapes) * sizeof(int)));
			checkcuda(cudaMalloc((void**)&maxNumZero_device, (dia_shapes) * sizeof(int)));
		}
	}

	int temp_sum_neg = 0;
	// prepare data on CPU
	for (int i = 0; i < dia_shapes; i++)
	{
		int j = shape_index[i];
		int neg = neg_format[j].neg;
		total_add_tasks += neg * (max(neg_format[j].end, neg_format[j].neg) - min(neg_format[j].start, neg_format[j].neg));
		
		end_add_tasks[i] = total_add_tasks;
		memcpy(&start_host[temp_sum_neg], neg_format[j].start, neg * sizeof(int));
		memcpy(&end_host[temp_sum_neg], neg_format[j].end, neg * sizeof(int));
		memcpy(&cstart_host[temp_sum_neg], neg_format[j].cStart, neg * sizeof(int));
		xStart_host[i] = neg_format[j].xStart;
		yStart_host[i] = neg_format[j].yStart;
		neg_host[i] = neg;

		if (size_negXmaxNumZero > 0)
		{
			memcpy(&neg_offsets_host[temp_sum_neg], neg_format[j].neg_offsets, neg * sizeof(int));
			memcpy(&numZeroNeg_host[temp_sum_neg], neg_format[j].numZeroNeg, neg * sizeof(int));
			for(int k = 0; k < neg; k ++){
				memcpy(&(rowZeroNeg_host[total_sub_tasks + k * neg_format[j].maxNumZero]), neg_format[j].rowZeroNeg[k], neg_format[j].maxNumZero * sizeof(int));
			}
			total_sub_tasks += neg * neg_format[j].maxNumZero;
			end_sub_tasks[i] = total_sub_tasks;
			maxNumZero_host[i] = neg_format[j].maxNumZero;
		}
		temp_sum_neg += neg;
	}
	
	// create cuda stream
	int num_streams = 2; // add, sub, ultra-sparse
	cudaStream_t *streams;
	checkcuda( cudaMallocHost((void**)&streams, num_streams * sizeof(cudaStream_t)));

	for(int i = 0; i < num_streams; i++){
		cudaStreamCreate(&streams[i]);
	}	

	// prepare y vector
	float *y_device, *x_device, *y_sub_device;
	checkcuda(cudaMalloc((void**)&x_device, N * sizeof(float)));
	checkcuda(cudaMemset(x_device, 0.0, N * sizeof(float)));
	checkcuda(cudaMalloc((void**)&y_device, M * sizeof(float))); 
	checkcuda(cudaMemset(y_device, 0.0, M * sizeof(float)));
	checkcuda(cudaMalloc((void**)&y_sub_device, (size_neg+1) * M * sizeof(float)));
	checkcuda(cudaMemset(y_sub_device, 0.0, (size_neg+1) * M * sizeof(float)));
	
	int block_num, thread_num;
	float time_add_and_sub = 0.0, time_gather=0.0;

	checkcuda(cudaMemcpy(x_device, xHostPtr, N * sizeof(float), cudaMemcpyHostToDevice)); // copy vector x

	// add and sub kernel
	if (dia_shapes > 0) {
		cudaEventRecord(start, 0);
		for(int t = 0; t < NUM_RUN; t++)
		{
			checkcuda(cudaMemset(y_sub_device, 0.0,  size_neg * M * sizeof(float)));

			checkcuda(cudaMemcpyAsync(start_device, start_host, size_neg * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
			checkcuda(cudaMemcpyAsync(end_device, end_host, size_neg * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
			checkcuda(cudaMemcpyAsync(cstart_device, cstart_host, size_neg * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
			checkcuda(cudaMemcpyAsync(end_add_tasks_device, end_add_tasks, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
			checkcuda(cudaMemcpyAsync(xStart_device, xStart_host, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
			checkcuda(cudaMemcpyAsync(yStart_device, yStart_host, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[0]));
			checkcuda(cudaMemcpyAsync(neg_device, neg_host, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[0]));

			thread_num = 256;
			block_num = total_add_tasks / thread_num + 1;
			SpMV_dia_add<<<block_num, thread_num, 0, streams[0]>>>( M, size_neg,
																	dia_shapes, 
																	xStart_device,
																	yStart_device,
																	neg_device,
																	end_add_tasks_device,
																	start_device, 
																	end_device, 
																	cstart_device, 
																	x_device, 
																	y_sub_device);

			if (size_negXmaxNumZero > 0) {
				checkcuda(cudaMemcpyAsync(neg_offsets_device, neg_offsets_host, size_neg * sizeof(int), cudaMemcpyHostToDevice, streams[1]));
				checkcuda(cudaMemcpyAsync(rowZeroNeg_device, rowZeroNeg_host, total_sub_tasks * sizeof(int), cudaMemcpyHostToDevice, streams[1]));
				checkcuda(cudaMemcpyAsync(numZeroNeg_device, numZeroNeg_host, size_neg * sizeof(int), cudaMemcpyHostToDevice, streams[1]));
				checkcuda(cudaMemcpyAsync(end_sub_tasks_device, end_sub_tasks, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[1]));
				checkcuda(cudaMemcpyAsync(xStart_device, xStart_host, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[1]));
				checkcuda(cudaMemcpyAsync(yStart_device, yStart_host, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[1]));
				checkcuda(cudaMemcpyAsync(neg_device, neg_host, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[1]));
				checkcuda(cudaMemcpyAsync(maxNumZero_device, maxNumZero_host, (dia_shapes) * sizeof(int), cudaMemcpyHostToDevice, streams[1]));

				thread_num = 256;
				block_num = total_sub_tasks / thread_num + 1;
				SpMV_dia_sub<<<block_num, thread_num, 0, streams[1]>>>( M, size_neg,
																		dia_shapes, 
																		xStart_device,
																		yStart_device,
																		neg_device,
																		maxNumZero_device, 
																		end_sub_tasks_device,
																		neg_offsets_device,
																		numZeroNeg_device,
																		rowZeroNeg_device, 
																		x_device, 
																		y_sub_device );
			}
		}
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0); //end timing
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&time_add_and_sub, start, stop); 
		time_add_and_sub /= NUM_RUN;
	}
	else {
		time_add_and_sub = 0.0;
	}
	printf("time_add_and_sub %.6f\n", time_add_and_sub);

	/* 
	FILE *fresult = fopen("TaiChi_CSR5_cuda_time.txt", "a+");
	if (fresult == NULL){printf("Create file failed.\n ");}
	fprintf(fresult, "%s time_add_and_sub %.6f ", matrixName, time_add_and_sub);
	fclose(fresult);
	*/

	// prepare array of ultra-sparse shape
	if (shape_index_len > dia_shapes)
	{
		int id_j = shape_index[shape_index_len-1];
		int sparse_nnz = neg_format[id_j].nnz;

		float * y_oth_host = (float*)malloc(M * sizeof(float));
		// ultra-sparse kernel
		call_anonymouslib(M, N, sparse_nnz, neg_format[id_j].csrRow, neg_format[id_j].csrCol, xHostPtr, y_oth_host, 1.0);
		
		checkcuda(cudaMemcpy(&y_sub_device[size_neg*M], y_oth_host, M * sizeof(float), cudaMemcpyHostToDevice));
		free(y_oth_host);
	}
	
	// gather y vectors
	if (dia_shapes == 0) {
		time_gather = 0.0;
	    checkcuda(cudaMemcpy(yHostPtr, &y_sub_device[size_neg*M], M * sizeof(float), cudaMemcpyDeviceToHost));
    }
	else {
		thread_num = 256;
		block_num = M / thread_num + 1;
		cudaEventRecord(start, 0);

        for (int i = 0; i < NUM_RUN; i++) {
            checkcuda(cudaMemset(y_device, 0.0, M * sizeof(float)));
		    SpMV_gather<<<block_num, thread_num,0>>>(y_sub_device, size_neg+1, M, y_device);
        }
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0); //end timing
		cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
		cudaEventElapsedTime(&time_gather, start, stop);    //计算时间差
		time_gather /= NUM_RUN;
        checkcuda(cudaMemcpy(yHostPtr, y_device, M * sizeof(float), cudaMemcpyDeviceToHost));
	}
	printf("time_gather %.6f\n", time_gather);
	
	/*
	fresult = fopen("TaiChi_CSR5_cuda_time.txt", "a+");
	if (fresult == NULL){printf("Create file failed.\n ");}
	fprintf(fresult, " time_gather %.6f\n", time_gather);
	fclose(fresult);
	*/

	// free resource
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkcuda(cudaFree(x_device));
	checkcuda(cudaFree(y_device));
	checkcuda(cudaFree(y_sub_device));

	if (dia_shapes > 0)
	{
		checkcuda(cudaFreeHost(start_host));
		checkcuda(cudaFreeHost(end_host));
		checkcuda(cudaFreeHost(cstart_host));

		checkcuda(cudaFree(start_device));
		checkcuda(cudaFree(end_device));
		checkcuda(cudaFree(cstart_device));
		checkcuda(cudaFree(end_add_tasks_device));
		checkcuda(cudaFree(xStart_device));
		checkcuda(cudaFree(yStart_device));
		checkcuda(cudaFree(neg_device));

		if (size_negXmaxNumZero > 0)
		{
		    checkcuda(cudaFreeHost(rowZeroNeg_host));
		    checkcuda(cudaFreeHost(neg_offsets_host));
			checkcuda(cudaFreeHost(numZeroNeg_host));
			
		    checkcuda(cudaFree(neg_offsets_device));
	 	    checkcuda(cudaFree(numZeroNeg_device));
		    checkcuda(cudaFree(rowZeroNeg_device));
			checkcuda(cudaFree(end_sub_tasks_device));
			checkcuda(cudaFree(maxNumZero_device));
		}
	}

	for(int i = 0; i < num_streams; i++){
		cudaStreamDestroy(streams[i]);
	}
	if (streams != NULL)
		checkcuda(cudaFreeHost(streams));

	cudaDeviceReset();
}
