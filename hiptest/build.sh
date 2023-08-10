if [[ "$HOSTNAME" = explorer ]]; then
  export ROCM_PATH=/opt/rocm-5.4.0
fi

make clean
make
make run-inline
