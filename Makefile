
all: test-inline test-outline

test-inline : inline.cu
	$(CUDA_PATH)/bin/nvcc inline.cu -o test-inline

kernel.ptx: kernel.cu
	$(CUDA_PATH)/bin/nvcc -ptx kernel.cu -o kernel.ptx

test-outline: kernel.ptx outline.c
	#$(CC) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -std=c99 outline.c -o test-outline  -lcuda  -lcudart
	$(CUDA_PATH)/bin/nvcc --compiler-options="-std=c99" -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 outline.c -o test-outline -lcuda -lcudart

run-inline: test-inline
	./test-inline

run-outline: test-outline
	./test-outline

debug-outline: test-outline
	gdb ./test-outline

clean:
	rm -f test-inline test-outline kernel.ptx

