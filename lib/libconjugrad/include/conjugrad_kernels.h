#ifndef CONJUGRAD_KERNELS_H
#define CONJUGRAD_KERNELS_H

#include "conjugrad.h"

#define CHECK_ERR(a) if (cudaSuccess != (a)) { printf("CUDA error No. %d in %s at line %d\n", a, __FILE__, __LINE__); exit(EXIT_FAILURE); }

#ifdef __cplusplus
extern "C" {
#endif

void vecnorm_gpu(
	conjugrad_float_t *d_g,
	conjugrad_float_t *d_res,
	int nvar
);

void vecdot_gpu(
	conjugrad_float_t *d_x,
	conjugrad_float_t *d_y,
	conjugrad_float_t *d_res,
	int nvar
);

void initialize_s_gpu(
	conjugrad_float_t *d_s,
	conjugrad_float_t *d_g,
	int nvar
);

void update_s_gpu(
	conjugrad_float_t *d_old_s,
	conjugrad_float_t *d_g,
	conjugrad_float_t beta,
	int nvar
);

void update_x_gpu(
	conjugrad_float_t *d_x,
	conjugrad_float_t *d_s,
	conjugrad_float_t alpha,
	conjugrad_float_t prevalpha,
	int nvar
);

#ifdef __cplusplus
}
#endif

#endif
