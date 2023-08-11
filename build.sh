#!/bin/bash
source /auto/software/iris/setup_system.source

export SM=sm_60
if [[ "$HOST" = "oswald00" ]]; then
  module load llvm/13.0.1
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda
fi
#  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda
#  #module load gnu/9.1.0 nvhpc/21.3 llvm-13.0.1-gcc-11.1.0-zmc5bxe
#  #export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda
#elif [[ "$HOST" = "leconte" ]]; then
#  module load gnu/9.2.0 nvhpc/21.3
#  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/cuda
#  export SM=sm_70
#elif [[ "$HOST" = equinox* ]]; then
#  module load gnu/11.1.0 nvhpc/22.5
#  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7
#elif [[ "$HOST" = whale* ]]; then
#  export CUDA_PATH=/usr/local/cuda
#else
#  export CUDA_PATH=/usr/local/cuda-11.4
#  export CPATH=/usr/local/cuda-11.4/include:$CPATH
#fi

#only update the path and library-path the first time!
if [[ $PATH != *$CUDA_PATH* ]]; then
  export PATH=$CUDA_PATH/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
fi

make clean
make
make run-inline
make run-outline
make run-inline-clang
make run-outline-clang
#make debug-outline
