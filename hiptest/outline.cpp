
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <signal.h>
#include <hip/hip_runtime.h>

size_t N = 10000000;
#define MAX_ERR 1e-6

#define HIP_CHECK(x)                                                          \
	do {                                                                   \
		hipError_t err = (x);                                         \
		if (err != hipSuccess) {                                     \
			fprintf(stderr, "HIP error: %s returned %d (%s) at %s:%d\n", \
				#x, err, hipGetErrorString(err), __FILE__, __LINE__);               \
			return err;                                            \
		}                                                              \
	} while (0)

void check_error(char* region_message){
 hipError_t err = hipGetLastError();
  if (err != HIP_SUCCESS){
    printf("ERROR! Failed to %s, with error code: %i\n",region_message,err);
    raise(SIGTERM);
  }
}
void host_vector_add(float *out, float *a, float *b, int n) {
  for(int i = 0; i < n; i++){
    out[i] = a[i] + b[i];
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
    buffer = (char*)malloc(length);
    if (buffer) {
      fread(buffer, 1, length, f);
    }
    fclose(f);
  }
  return buffer;
}

int main(int argc, char* argv[]){
  if (argc < 2) {
    fprintf(stderr, "Usage: %s kernel.hip", argv[0]);
  }
  char *kernel_filename = argv[1];
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device,0));
  hipCtx_t ctx;
  HIP_CHECK(hipCtxCreate(&ctx,0,device));
  hipModule_t mod;
  HIP_CHECK(hipModuleLoad(&mod, kernel_filename));
  hipFunction_t vector_add;
  HIP_CHECK(hipModuleGetFunction(&vector_add, mod, "vector_add"));
  check_error((char*)"load hip kernel");

  int runtime_version;
  HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
  printf("HIP runtime version: %d\n", runtime_version);

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
  HIP_CHECK(hipMalloc((void**)&d_a, sizeof(float) * N));
  HIP_CHECK(hipMalloc((void**)&d_b, sizeof(float) * N));
  HIP_CHECK(hipMalloc((void**)&d_out, sizeof(float) * N));

  // Transfer data from host to device memory
  HIP_CHECK(hipMemcpy(d_a, a, sizeof(float) * N, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, b, sizeof(float) * N, hipMemcpyHostToDevice));
  //hipMemcpy(d_out, out, sizeof(float) * N, hipMemcpyHostToDevice);

  // Main function
  host_vector_add(out, a, b, N);
  //vector_add<<<1,1>>>(d_out, d_a, d_b, N);
  void* args[] = {&d_out, &d_a, &d_b, &N};
  hipModuleLaunchKernel(vector_add, 1,1,1, 1,1,1, 0,0, (void**)args, NULL);

  check_error((char*)"launching vector_add hip kernel");

  float* test_out = (float*)malloc(sizeof(float) * N);
  HIP_CHECK(hipMemcpy(test_out, d_out, sizeof(float) * N, hipMemcpyDeviceToHost));

  // validate results (on the host)
  for(int i = 0; i < N; i++){
    if (fabs(out[i] - test_out[i]) >= MAX_ERR){
      printf("ERROR: results don't match!\n");
      printf("index: %i host: %f device: %f\n",i,out[i],test_out[i]);
      return(EXIT_FAILURE);
    }
  }

  printf("Inline Check Passed --- the CUDA runtime can correctly run kernels.\n");

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_out);

  free(a);
  free(b);
  free(out);
  free(test_out);
}

