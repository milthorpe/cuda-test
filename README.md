# cuda-test

A simple test of the CUDA runtime. It performs vector addition and checks the numerical result. There is an "inline" and an "outline" version, the former compiles the kernel in the host-code (single binary), while the latter loads PTX at runtime.