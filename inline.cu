#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
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

__global__ void vector_add(float *out, float *a, float *b, int n) {
  for(int i = 0; i < n; i++){
    out[i] = a[i] + b[i];
  }
}

int main(){
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
  CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc((void**)&d_b, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(float) * N));

  // Transfer data from host to device memory
  CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));
  //cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Main function
  host_vector_add(out, a, b, N);

  vector_add<<<1,1>>>(d_out, d_a, d_b, N);

  float* test_out = (float*)malloc(sizeof(float) * N);
  CUDA_CHECK(cudaMemcpy(test_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost));

  // validate results (on the host)
  for(int i = 0; i < N; i++){
    if (fabs(out[i] - test_out[i]) >= MAX_ERR){
      printf("ERROR: results don't match!\n");
      printf("index: %i host: %f device: %f\n",i,out[i],test_out[i]);
      return(EXIT_FAILURE);
    }
  }

  printf("Inline Check Passed --- the CUDA runtime can correctly run kernels.\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  free(a);
  free(b);
  free(out);
  free(test_out);
}

