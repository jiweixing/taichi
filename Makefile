all: spmv

#GLOBAL_PARAMETERS
VALUE_TYPE = float
NUM_RUN = 1000

#CUDA_PARAMETERS
ARCH = 70
CUDA_CC = nvcc
NVCC_FLAGS = -O3  -w -m64 -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDA_INSTALL_PATH = /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I/home/gjh/NVIDIA_CUDA-10.0_Samples/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse

TaiChi.o: TaiChi.cu
	$(CUDA_CC) $(NVCC_FLAGS) -o TaiChi.o -c TaiChi.cu $(CUDA_INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
main.o: main.cpp
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o main.o -c main.cpp $(CUDA_INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
spmv: TaiChi.o main.o
	$(CUDA_CC) $(NVCC_FLAGS) TaiChi.o main.o -o spmv $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)

clean:
	rm *.o 
