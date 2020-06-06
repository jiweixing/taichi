#include <cuda_runtime.h>
#include "cusparse.h"

// record the basic info of each shape
typedef struct shape{
	int id;
	char format[20];        // format for each submatrix
    char category[1024];	// rectangular,triangular, or diagonal
	int x1;					// row index of left upper corner in thumbnail
	int y1;					// column index of left upper corner in thumbnail
	int a1;					// long of the shape in thumbnail
	int b1;					// width of the shape in thumbnail
	int area1;				// number of non-zero pixels of the shape in thumbnail
	int x;					// row index of left upper corner in matrix 
	int y;					// column index of left upper corner in matrix 
	int a;                  // long of the shape in matrix
	int b;                  // width of the shape in matrix
	int *cooRow;
	int *csrCol;
	float *cooVal;
	int nnz_submatrix;
	int width;
}shape;

typedef struct TaiChi {
	char shape; 	//	0-diagnal 1-ultra-sparse 2-triangle-or-rectangle
	int M_sub;		// row length of submatrix
	int N_sub;		// column length of submatrix
	int xStart;		// row offset of submatrix
	int yStart;		// column length of submatrix
	int width;		// diagonal width
		
	// for dense diagnals
	int neg;				// number of dense diagonals
	int maxNumZero;			// maximal number of zeros of dense diagonal 
	int *neg_offsets;		// offset of dense diagonal	
	int *numZeroNeg;		// number of zeros for each dense diagonal
	int **rowZeroNeg;		// row index of zeros in each dense diagonal

	int *start;
	int *end;
	int *cStart;

	// sparse diagonal
	// int pos;				
	// int maxNumNnz;			
	// int *pos_offsets;		
	// int *numNnzPos;			
	// int **rowNnzPos;		

	// for others
	int nnz;
	int *csrRow;
	int *csrCol;
} TaiChi;

#ifndef VALUE_TYPE
#define VALUE_TYPE float
#endif

#ifndef NUM_RUN
#define NUM_RUN 1000
#endif

#define ZERO 1e-8

int readMtx(char *filename, int &m, int &n, int &nnzA, int *&csrRowPtrA, int *&csrColIdxA,
	float *&csrValA);
void partition_shapes(FILE *fSdf, int M, int N, int nnz, int* &csrRowIndexHostPtr, int* &csrColIndexHostPtr, float* &csrValHostPtr, int mF, int nF, int num_shapes, shape *h, TaiChi *neg_format);
void cal_shape_size(int M, int N, int mF, int x1, int y1, int a1, int b1, int &x, int &y, int &a, int &b, float &gain);
int cal_memory(int num_shapes, TaiChi *neg_format);
void partition_dia_shape(int M, int N, int nnz, int* &csrRowIndexHostPtr, int* &csrColIndexHostPtr, float* &csrValHostPtr, float gain, shape &hi, TaiChi &neg_format_i);
void spmv_stream_gpu(int M, int N, TaiChi *neg_format, int num_shapes, float *xHostPtr, float *yHostPtr);
void queryDevice();
inline void checkcuda(cudaError_t result);
inline void checkcusparse(cusparseStatus_t result);
int max(int *a, int len);
int min(int *a, int len);
void cal_start_end (int M_sub, int N_sub, int width, int dia_offsets_temp, int n, int &start_temp, int &end_temp);
int call_anonymouslib(int m, int n, int nnzA,
	int *csrRowPtrA, int *csrColIdxA,
	VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha);
