#include "conjugrad.h"

#ifndef HELP_FUNCTIONS_H
#define HELP_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

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
);

#ifdef __cplusplus
}
#endif

#endif
