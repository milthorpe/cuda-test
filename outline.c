#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <signal.h>
#include <cuda.h>
#include <cuda_runtime.h>

size_t N = 10000000;
#define MAX_ERR 1e-6

#define CUDA_CHECK(x)                                                          \
	do {                                                                   \
		cudaError_t err = (x);                                         \
		if (err != CUDA_SUCCESS) {                                     \
			fprintf(stderr, "CUDA error: %s returned %d (%s) at %s:%d\n", \
				#x, err, cudaGetErrorString(err), __FILE__, __LINE__);               \
			return err;                                            \
		}                                                              \
	} while (0)

void host_vector_add(float *out, float *a, float *b, int n) {
  for(int i = 0; i < n; i++){
    out[i] = a[i] + b[i];
  }
}

void check_error(char* region_message){
 cudaError_t err = cudaGetLastError();
  if (err != CUDA_SUCCESS){
    printf("ERROR! Failed to %s, with error code: %i\n",region_message,err);
    raise(SIGTERM);
  }
}

char *read_file(char *filename) {
	char *buffer = 0;
	long length;
	FILE *f = fopen(filename, "rb");

	if (f) {
		fseek(f, 0, SEEK_END);
		length = ftell(f);
		fseek(f, 0, SEEK_SET);
		buffer = malloc(length);
		if (buffer) {
			fread(buffer, 1, length, f);
		}
		fclose(f);
	}
	return buffer;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s kernel.ptx", argv[0]);
  }
  char *kernel_filename = argv[1];
  CUDA_CHECK(cuInit(0));
  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device,0));
  CUcontext ctx;
  CUDA_CHECK(cuCtxCreate(&ctx,0,device));
  CUmodule mod;
  CUDA_CHECK(cuModuleLoad(&mod, kernel_filename));
  //char *kernel_data = read_file("kernel.ptx");
  //printf("loaded kernel:\n%s\n", kernel_data);
  //CUDA_CHECK(cuModuleLoadData(&mod, kernel_data));
  //free(kernel_data);
  CUfunction vector_add;
  CUDA_CHECK(cuModuleGetFunction(&vector_add, mod, "vector_add"));
  check_error("load cuda kernel");

  float *a, *b, *out, *d_a, *d_b, *d_out; 

  // Allocate memory
  a   = (float*)malloc(sizeof(float) * N);
  b   = (float*)malloc(sizeof(float) * N);
  out = (float*)malloc(sizeof(float) * N);

  // Initialize array
  for(int i = 0; i < N; i++){
    a[i] = 1.0f; b[i] = 2.0f;
  }

  // Allocate device memory for a
  cudaMalloc((void**)&d_a, sizeof(float) * N);
  cudaMalloc((void**)&d_b, sizeof(float) * N);
  cudaMalloc((void**)&d_out, sizeof(float) * N);
  check_error("allocating cuda buffers");

  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);
  check_error("transferring cuda memory");

  // Main function
  host_vector_add(out, a, b, N);
  // vector_add<<<1,1>>>(d_out, d_a, d_b, N);
  void* args[] = {&d_out,&d_a,&d_b,&N};
  cuLaunchKernel(vector_add, 1,1,1, 1,1,1, 0,0, (void**)args, NULL);
  check_error("launching vector_add cuda kernel");

  float* test_out = (float*)malloc(sizeof(float) * N);
  cudaMemcpy(test_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
  check_error("transferring cuda memory---collecting result");

  // validate results (on the host)
  for(int i = 0; i < N; i++){
    if (fabs(out[i] - test_out[i]) >= MAX_ERR){
      printf("ERROR: results don't match!\n");
      printf("index: %i host: %f device: %f\n",i,out[i],test_out[i]);
      return(EXIT_FAILURE);
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  check_error("freeing cuda memory");

  cuCtxDestroy(ctx);
  printf("Outline Check Passed --- the CUDA runtime can correctly load and run ptx files.\n");

  free(a);
  free(b);
  free(out);
  free(test_out);
}

