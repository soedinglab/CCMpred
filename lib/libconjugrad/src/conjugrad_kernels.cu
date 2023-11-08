#ifdef __cpluplus
extern "C" {
#endif
#include "conjugrad_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#ifdef __cpluplus
}
#endif

//forward declaration of device functions
__device__ void sum_reduction_function(
	volatile conjugrad_float_t *s_data,
	int tid
);

__device__ void warp_reduction(
	volatile conjugrad_float_t *s_data,
	int tid
);

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

__global__ void vecdot_inter(
	conjugrad_float_t *d_x,
	conjugrad_float_t *d_y,
	conjugrad_float_t *d_vecdot_inter,
	int nvar
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockDim.x * gridDim.x;

	extern __shared__ conjugrad_float_t s_vecdot_inter[];

	s_vecdot_inter[threadIdx.x] = F0; // 0.0

	while (i < nvar) {
		s_vecdot_inter[threadIdx.x] = d_x[i] * d_y[i] + s_vecdot_inter[threadIdx.x];
		i += offset;
	}
	__syncthreads();

	sum_reduction_function(s_vecdot_inter, threadIdx.x);

	if (0 == threadIdx.x) {
		d_vecdot_inter[blockIdx.x] = s_vecdot_inter[0];
	}
}

__global__ void update_s(
	conjugrad_float_t *d_old_s,
	conjugrad_float_t *d_g,
	conjugrad_float_t beta,
	int nvar
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockDim.x * gridDim.x;

	while (i < nvar) {
		d_old_s[i] = beta * d_old_s[i] - d_g[i];
		i += offset;
	}
}

__global__ void initialize_s(
	conjugrad_float_t *d_s,
	conjugrad_float_t *d_g,
	int nvar) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockDim.x * gridDim.x;

	while (i < nvar) {
		d_s[i] = - d_g[i];
		i += offset;
	}
}

__global__ void update_x(
	conjugrad_float_t *d_x,
	conjugrad_float_t *d_s,
	conjugrad_float_t alpha,
	conjugrad_float_t prevalpha,
	int nvar) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int offset = blockDim.x * gridDim.x;

	while (i < nvar) {
		d_x[i] = (alpha - prevalpha) * d_s[i] + d_x[i];
		i += offset;
	}
}

#ifdef __cpluplus
extern "C" {
#endif
void vecnorm_gpu(
	conjugrad_float_t *d_x,
	conjugrad_float_t *res,
	int nvar
) {
	unsigned int nblocks = 128;
	unsigned int nthreads = 256;
	size_t nbytes = sizeof(conjugrad_float_t) * nthreads;

	conjugrad_float_t *d_inter;
	conjugrad_float_t *d_res;
	CHECK_ERR(cudaMalloc((void **) &d_inter, sizeof(conjugrad_float_t) * nblocks));
	CHECK_ERR(cudaMalloc((void **) &d_res, sizeof(conjugrad_float_t)));

	vecdot_inter<<<nblocks, nthreads, nbytes>>>(d_x, d_x, d_inter, nvar);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	CHECK_ERR(err);


	nbytes = sizeof(conjugrad_float_t) * nblocks;
	sum_reduction<<<1, nblocks, nbytes>>>(d_inter, d_res, nblocks); 
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CHECK_ERR(err);


	CHECK_ERR(cudaMemcpy(res, d_res, sizeof(conjugrad_float_t), cudaMemcpyDeviceToHost));

	CHECK_ERR(cudaFree(d_inter));
	CHECK_ERR(cudaFree(d_res));
}

void vecdot_gpu(
	conjugrad_float_t *d_x,
	conjugrad_float_t *d_y,
	conjugrad_float_t *res,
	int nvar
) {
	unsigned int nblocks = 128;
	unsigned int nthreads = 256;
	size_t nbytes = sizeof(conjugrad_float_t) * nthreads;

	conjugrad_float_t *d_inter;
	conjugrad_float_t *d_res;
	CHECK_ERR(cudaMalloc((void **) &d_inter, sizeof(conjugrad_float_t) * nblocks));
	CHECK_ERR(cudaMalloc((void **) &d_res, sizeof(conjugrad_float_t)));

	vecdot_inter<<<nblocks, nthreads, nbytes>>>(d_x, d_y, d_inter, nvar);
	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());

	nbytes = sizeof(conjugrad_float_t) * nblocks;
	sum_reduction<<<1, nblocks, nbytes>>>(d_inter, d_res, nblocks); 
	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());

	CHECK_ERR(cudaMemcpy(res, d_res, sizeof(conjugrad_float_t), cudaMemcpyDeviceToHost));

	CHECK_ERR(cudaFree(d_inter));
	CHECK_ERR(cudaFree(d_res));
}

void initialize_s_gpu(
	conjugrad_float_t *d_s,
	conjugrad_float_t *d_g,
	int nvar
) {
	int nblocks = 128;
	int nthreads = 256;

	initialize_s<<<nblocks, nthreads>>>(d_s, d_g, nvar);
	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}

void update_s_gpu(
	conjugrad_float_t *d_old_s,
	conjugrad_float_t *d_g,
	conjugrad_float_t beta,
	int nvar
) {
	unsigned int nblocks = 128;
	unsigned int nthreads = 256;

	update_s<<<nblocks, nthreads>>>(d_old_s, d_g, beta, nvar);
	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}

void update_x_gpu(
	conjugrad_float_t *d_x,
	conjugrad_float_t *d_s,
	conjugrad_float_t alpha,
	conjugrad_float_t prevalpha,
	int nvar
) {
	int nblocks = 128;
	int nthreads = 256;
	update_x<<<nblocks, nthreads>>>(d_x, d_s, alpha, prevalpha, nvar);
	cudaDeviceSynchronize();
	CHECK_ERR(cudaGetLastError());
}

#ifdef __cpluplus
}
#endif
