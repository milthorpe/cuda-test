if [ "$HOSTNAME" = "oswald03" ]; then
  module load gnu/9.2.0 nvhpc/21.7
	export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda
elif [ "$HOSTNAME" = "leconte" ]; then
  module load gnu/9.2.0 nvhpc/21.3
  export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/cuda
fi

#only update the path and library-path the first time!
if [[ $PATH != *$CUDA_PATH* ]]; then
  export PATH=$CUDA_PATH/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
fi

make clean
make
make run-inline
make run-outline
#make debug-outline
