# GPU-accelerated filters
This is a parallel implementation of xor and fuse filter construction. The code is tested with CUDA 10.x versions.

# Running the code
```bash
# Fuse filter
$> nvcc -o fusebench fusebench.cu -O3
$> ./fusebench

# Xor filter
$> nvcc -o xorbench bench.cu -O3
$> ./xorbench
```

# Acknowledgements
This implementation builds on sequential xor and fuse filter constructions by Graf and Lemire http://https://github.com/FastFilter/xor_singleheader