#include "help_functions.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void sum_reduction(
	conjugrad_float_t *d_in,
	conjugrad_float_t *d_out,
	int nvar
) {
	int i = threadIdx.x;

	extern __shared__ conjugrad_float_t s_data[];

	s_data[threadIdx.x] = F0; // 0.0

	while (i < nvar) {
		s_data[threadIdx.x] += d_in[i];
		i += blockDim.x;
	}
	__syncthreads();

	sum_reduction_function(s_data, threadIdx.x);

	if (0 == threadIdx.x) {
		d_out[0] = s_data[0];
	}
}

__device__ void sum_reduction_function(
	volatile conjugrad_float_t *s_data,
	int tid
) {
	for (unsigned int s = blockDim.x >> 1; s > 32; s >>= 1) {
		if (threadIdx.x < s) {
			s_data[threadIdx.x] += s_data[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x < 32) {
		warp_reduction(s_data, threadIdx.x);
	}
}

__device__ void warp_reduction(
	volatile conjugrad_float_t *s_data,
	int tid
) {
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid +  8];
	s_data[tid] += s_data[tid +  4];
	s_data[tid] += s_data[tid +  2];
	s_data[tid] += s_data[tid +  1];
}

#ifdef __cplusplus
}
#endif
