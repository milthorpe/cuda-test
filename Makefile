
all: test-inline test-outline test-inline-clang

test-inline : inline.cu
	$(CUDA_PATH)/bin/nvcc inline.cu -o test-inline

kernel.ptx: kernel.cu
	$(CUDA_PATH)/bin/nvcc -ptx kernel.cu -o kernel.ptx

test-outline: kernel.ptx outline.c
	#$(CC) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -std=c99 outline.c -o test-outline  -lcuda  -lcudart
	$(CUDA_PATH)/bin/nvcc --compiler-options="-std=c99" -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 outline.c -o test-outline -lcuda -lcudart

test-inline-clang: inline.cu
	clang++ -O2 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_60 inline.cu -L$(CUDA_PATH)/lib64 -lcudart -o test-inline-clang

run-inline: test-inline
	./test-inline

run-outline: test-outline
	./test-outline

run-inline-clang: test-inline-clang
	./test-inline-clang

debug-outline: test-outline
	gdb ./test-outline

clean:
	rm -f test-inline test-outline kernel.ptx test-inline-clang

