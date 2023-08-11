
all: test-inline test-outline test-inline-clang

test-inline : inline.cu
	$(CUDA_PATH)/bin/nvcc inline.cu -o test-inline

kernel.ptx: kernel.cu
	$(CUDA_PATH)/bin/nvcc -ptx kernel.cu -o kernel.ptx

test-outline: kernel.ptx outline.c
	$(CUDA_PATH)/bin/nvcc --compiler-options="-std=c99" -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 outline.c -o test-outline -lcuda -lcudart

test-inline-clang: inline.cu
	clang++ -O2 --cuda-path=$(CUDA_PATH) inline.cu -L$(CUDA_PATH)/lib64 -lcudart -o test-inline-clang

kernel-clang.ptx: kernel.cu
	clang++ -S -emit-llvm --cuda-gpu-arch=$(SM) kernel.cu
	llc -mcpu=$(SM) kernel-cuda-nvptx64-nvidia-cuda-$(SM).ll -o kernel-clang.ptx

test-outline-clang: kernel-clang.ptx outline.c
	clang -I$(CUDA_PATH)/include outline.c -L$(CUDA_PATH)/lib64 -lcuda -lcudart -o test-outline-clang

run-inline: test-inline
	./test-inline

run-outline: test-outline
	./test-outline kernel.ptx

run-inline-clang: test-inline-clang
	./test-inline-clang

run-outline-clang: test-outline-clang
	./test-outline-clang kernel-clang.ptx

debug-outline: test-outline
	gdb ./test-outline

.PHONY:	clean

clean:
	rm -f test-inline test-outline kernel.ptx test-inline-clang

