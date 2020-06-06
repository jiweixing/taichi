#include <stdio.h>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cfloat>
#include <errno.h>
#include "mmio.h"
#include "TaiChi.h"
#include <dirent.h>

using namespace std;

/*** Declaration ***/
int M, N, nnz;	
char matrixName[1024] = {0};

int *csrRowIndexHostPtr = 0;
int *csrColIndexHostPtr = 0;
float * csrValHostPtr = 0;
float * xHostPtr = 0;
float * yHostPtr = 0;

int main(int argc, char *argv[]){
	if (argc != 3)
	{
		printf("usage: main MatrixPath sdfPath\n");
		return 0;
	}

	printf("%s %s %s\n", argv[0], argv[1], argv[2]);

	queryDevice();

	char matrix_dir[1024] = {0}, sdf_dir[1024] = {0};
	
	strcpy(matrix_dir, argv[1]);
	strcpy(sdf_dir, argv[2]);

	//find matrix file and sdf file
    DIR *matrix_dir_handle, *sdf_dir_handle;
	struct dirent *matrix_entry;
	
    matrix_dir_handle = opendir(matrix_dir);
    sdf_dir_handle = opendir(sdf_dir);

	int counter = 0;
	int counter_wrong = 0;
    while ((matrix_entry = readdir(matrix_dir_handle)) != NULL && ((readdir(sdf_dir_handle)) != NULL))        
    {
        if(strcmp(matrix_entry->d_name,"..") != 0 && strcmp(matrix_entry->d_name,".") != 0){
			//decide if the matrix name matches the sdf name
			string matrixPureName(matrix_entry->d_name), sdfPureName;
			matrixPureName.erase(matrixPureName.end()-4, matrixPureName.end());
            sdfPureName = matrixPureName + ".txt";
			
			// read matrix
			char source[1024] = {0};// full pathname of matrix
			strcpy(source, argv[1]);
			strcat(source, "/");
			strcat(source, matrix_entry->d_name);
			strcpy(matrixName, matrix_entry->d_name);
			printf("Matrix: %s\n",matrixName);
			readMtx(source, M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr);
			printf("M=%d N=%d nnz=%d\n", M, N, nnz);

			xHostPtr = (float*)malloc(sizeof(float) * N);
			for (int i = 0; i < N; i++) xHostPtr[i] = 1.0;

			// Calculate accurate result
			float * y_ref = (float *)malloc(sizeof(float) * M);
			for (int i = 0; i < M; i++) {
                y_ref[i] = 0.0;
				for (int j = csrRowIndexHostPtr[i]; j < csrRowIndexHostPtr[i+1]; j++) {
					y_ref[i] += csrValHostPtr[j] * xHostPtr[csrColIndexHostPtr[j]];
				}
			}

			// read sdf
			int mF = 0, nF = 0, num_shapes = 0; //number of rows,cols and pixels of the thumbnail
			FILE* fSdf = NULL;
			char sdfFile[1024] = {0};
			strcpy(sdfFile, argv[2]);
			strcat(sdfFile, "/");
			strcat(sdfFile, sdfPureName.c_str());
			fSdf = fopen(sdfFile, "r");
			if (fSdf == NULL) {printf("Open %s file failed.\n ", sdfFile);}
			fscanf(fSdf, "%d %d %d", &mF, &nF, &num_shapes);
			
            shape* h = (shape*)malloc( (num_shapes+1) * sizeof(shape));
			memset(h, 0, (num_shapes+1) * sizeof(shape));

			TaiChi * neg_format = (TaiChi*)malloc(sizeof(TaiChi)*(num_shapes+1));
			memset(neg_format, 0, sizeof(TaiChi)*(num_shapes+1));
			
			// partition shapes
			printf("Start Partitioning shapes.\n");
			partition_shapes(fSdf, M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr, mF, nF, num_shapes, h, neg_format);
			fclose(fSdf);
			printf("End Partitioning shapes.\n");


			// if existing dense diagonals, then we can first free these three arrays
			if (neg_format[num_shapes].nnz < nnz) {
				if (csrRowIndexHostPtr != NULL) { free(csrRowIndexHostPtr); csrRowIndexHostPtr = NULL; }
				if (csrColIndexHostPtr != NULL) { free(csrColIndexHostPtr); csrColIndexHostPtr = NULL; }
			}
			if (csrValHostPtr != NULL) { free(csrValHostPtr); csrValHostPtr = NULL; }

			// calculate the non-zeros rate 
			int sum_sub_nnz = 0;
			for(int i=0; i < num_shapes; i++){ 
				sum_sub_nnz += neg_format[i].nnz;
			}
			
			float rate = (float)sum_sub_nnz / (float)nnz;
			yHostPtr = (float*)malloc(sizeof(float) * M);
			memset(yHostPtr, 0.0, sizeof(float) * M);

			// SpMV
			printf("\nBegain call spmv function.\n");
			spmv_stream_gpu(M, N, neg_format, num_shapes, xHostPtr, yHostPtr);
			printf("End call spmv function.\n\n");

            // validate calculated result
			for (int i = 0; i < M; i++) {
				if ( abs(yHostPtr[i] - y_ref[i]) > 1e-6 ) {
					counter_wrong ++;
				}
			}

			//free memory
			if (xHostPtr != NULL) { free(xHostPtr); xHostPtr = NULL;}
			if (yHostPtr != NULL) { free(yHostPtr); yHostPtr = NULL;}
			if (y_ref != NULL) { free(y_ref); y_ref = NULL;}
							
			if (h != NULL) {
				free(h);
				h = NULL;
			}
			
			for(int i = 0; i < num_shapes; i++)
			{
				if(neg_format[i].neg_offsets != NULL)
				{
					free(neg_format[i].neg_offsets);
					neg_format[i].neg_offsets = NULL;
				}
				if(neg_format[i].numZeroNeg != NULL)
				{
					free(neg_format[i].numZeroNeg);
					neg_format[i].numZeroNeg = NULL;
				}
				for(int d = 0; d < neg_format[i].neg; d++)
				{
					if (neg_format[i].rowZeroNeg != NULL) {
						if(neg_format[i].rowZeroNeg[d] != NULL)
						{
							free(neg_format[i].rowZeroNeg[d]);
							neg_format[i].rowZeroNeg[d] = NULL;
						}
					}
				}
				if(neg_format[i].rowZeroNeg != NULL)
				{
					free(neg_format[i].rowZeroNeg);
					neg_format[i].rowZeroNeg = NULL;
				}
                printf("free over neg_format[%d]\n", i);
			}


			if(neg_format[num_shapes].csrRow != NULL)
			{
				free(neg_format[num_shapes].csrRow);
				neg_format[num_shapes].csrRow = NULL;
			}
			if(neg_format[num_shapes].csrCol != NULL)
			{
				free(neg_format[num_shapes].csrCol);
				neg_format[num_shapes].csrCol = NULL;
			}

			if(neg_format != NULL)
			{
				free(neg_format);
				neg_format = NULL;
			}

			csrRowIndexHostPtr = NULL;
			csrColIndexHostPtr = NULL;

			counter++;
			printf("%d finished, %d error\n",counter, counter_wrong);
            // if (counter_wrong > 0) break;
		}// end if     
    }//end while

	printf("\n--------------------------------------------------\n");
	printf("\n finished! %d matrices tested, %d error\n", counter, counter_wrong);
	return 0;
}
