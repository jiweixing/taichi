# TaiChi-based SoMV

TaiChi is a hybrid format for binary sparse matrix and fully exploits the data distribution of non-zero elements. Input matrices are firstly partitioned into relative dense and ultra-sparse areas, then the dense areas are encoded inversely by marking “0”s, while the ultra-sparse area is encoded using popular CSR5 format by marking “1”s. We also design a new SpMV algorithm just using addition and subtraction for binary matrices based on our partition and encoding format.

## Usage
- change the related path in Makefile
- `make`
- `./spmv matrixPath sdfPath`
  - **matrixPath** is the directory of binary sparse matrices, which require to be downloaded by yourself from the SuiteSparse Matrix Collection. We uploaded some example matrices to `dataset/matrixPath/`.
  - unzip the `dataset/sdfFile.zip`, and **sdfPath** is the directory of sdf file: `./dataset/sdfFile/`

