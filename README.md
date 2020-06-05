# TaiChi-based SoMV

TaiChi is a hybrid format for binary sparse matrix and fully exploits the data distribution of non-zero elements. Input matrices are firstly partitioned into relative dense and ultra-sparse areas, then the dense areas are encoded inversely by marking “0”s, while the ultra-sparse area is encoded using popular CSR5 format by marking “1”s. We also design a new SpMV algorithm just using addition and subtraction for binary matrices based on our partition and encoding format. We evaluate our SpMV algorithm using some real-world binary sparse matrices from the SuiteSparse Matrix Collection. Evaluation results show that the speedup of CSR5-NV-based SpMV to CSR5-based SpMV is up to 3.26x on GTX 1080 Ti and 2.20x on Tesla V100-SXM2, and our hybrid encoding for binary matrix significantly compresses the original matrix and obtains the highest speedup of 6.89x and 4.21x on GTX 1080 Ti and Tesla V100-SXM2 respectively.

## Environment Information
We run the script `https://github.com/SC-Tech-Program/Author-Kit/blob/master/collect_environment.sh` and collect the output to the file `environment_info.txt`, including OS, compilers, CPU, GPU, memory, disk, and so on.

## Usage
- change the CUDA path in Makefile
- change the ARCH: 61 in GTX 1080 Ti, 70 in Telsa V100-SXM2.
- `make`
- `./spmv matrixPath sdfPath`
  - **matrixPath** is the directory of binary sparse matrices, which require to be downloaded by yourself from the SuiteSparse Matrix Collection. We uploaded some example matrices to `dataset/matrixPath/`.
  - unzip the `dataset/sdfFile.zip`, and **sdfPath** is the directory of sdf file: `./dataset/sdfFile/`

## Timing
The timing codes of different parts are included in the program, the total SpMV time should be equal to the sum of `time_add_and_sub`, `CSR5_transfer`, `CSR5-based SpMV time`, and `time_gather`.
