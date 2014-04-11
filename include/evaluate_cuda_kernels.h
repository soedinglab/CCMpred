#ifndef EVALUATE_CUDA_KERNELS_H
#define EVALUATE_CUDA_KERNELS_H

#include <stdio.h>
#include "conjugrad.h"
#include "evaluate_cuda.h"
#include "ccmpred.h"

#define CHECK_ERR(err) {if (cudaSuccess != (err)) { printf("CUDA error No. %d in %s at line %d\n", (err), __FILE__, __LINE__); exit(EXIT_FAILURE); } }

#ifdef __cplusplus
extern "C" {
#endif

// C function calls go here
void gpu_pc(
	const unsigned char *d_msa,
	const conjugrad_float_t *d_x1,
	const conjugrad_float_t *d_x2,
	conjugrad_float_t *d_precompiled,
	conjugrad_float_t *d_precompiled_sum,
	conjugrad_float_t *d_precompiled_norm,
	const conjugrad_float_t *d_weights,
	const int ncol,
	const int nrow
);

void gpu_compute_fx(
	const conjugrad_float_t *d_precompiled,
	const conjugrad_float_t *d_precompiled_sum,
	const int ncol,
	const int nrow,
	conjugrad_float_t *d_fx
);

void gpu_compute_regularization(
	conjugrad_float_t *d_g,
	conjugrad_float_t lambda_single,
	conjugrad_float_t lambda_pair,
	conjugrad_float_t *d_reg,
	conjugrad_float_t *d_histograms,
	int nvar,
	int nsingle
);

void gpu_compute_node_gradients(
	conjugrad_float_t *d_g1,
	conjugrad_float_t *d_precompiled_norm,
	int ncol,
	int nrow
);

void gpu_compute_edge_gradients(
	unsigned char *d_msa_transposed,
	conjugrad_float_t *d_g2,
	conjugrad_float_t *d_precompiled_norm,
	int ncol,
	int nrow
);

void gpu_tranpose_msa(
	unsigned char *d_msa,
	unsigned char *d_msa_transposed,
	int ncol,
	int nrow
);

void gpu_initialize_histograms_weighted(
	unsigned char *d_msa,
	conjugrad_float_t *d_histograms,
	const conjugrad_float_t *d_weights,
	int ncol,
	int nrow
);

void gpu_compute_weights_simple(
	conjugrad_float_t *d_weights,
	conjugrad_float_t threshold,
	unsigned char *d_msa,
	int nrow,
	int ncol
);

#ifdef __cplusplus
}
#endif

#endif
