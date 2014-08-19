#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "sequence.h"

#include "evaluate_cuda_kernels.h"

/* Declare global GPU pointers */
unsigned char *d_msa;
unsigned char *d_msa_transposed;
conjugrad_float_t *d_precompiled;
conjugrad_float_t *d_precompiled_sum;
conjugrad_float_t *d_precompiled_norm;
conjugrad_float_t *d_histograms;
conjugrad_float_t *d_weights;

conjugrad_float_t evaluate_cuda (
	void *instance,
	const conjugrad_float_t *d_x_padded,
	conjugrad_float_t *d_g_padded,
	const int new_nvar
) {

	userdata *ud = (userdata *)instance;
	const int ncol = ud->ncol;
	const int nrow = ud->nrow;
	const int nsingle = ud->nsingle;
	int nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);
	const int nvar = nsingle + ncol * ncol * N_ALPHA * N_ALPHA;

	const conjugrad_float_t *d_x1_padded = d_x_padded;
	const conjugrad_float_t *d_x2_padded = &d_x_padded[nsingle_padded];
	

	CHECK_ERR(cudaMemcpy(d_g_padded, d_x_padded, sizeof(conjugrad_float_t) * new_nvar, cudaMemcpyDeviceToDevice));
	conjugrad_float_t *d_g1_padded = d_g_padded;
	conjugrad_float_t *d_g2_padded = &d_g_padded[nsingle_padded];


	gpu_pc((const unsigned char *)d_msa, d_x1_padded, d_x2_padded, d_precompiled, d_precompiled_sum, d_precompiled_norm, (const conjugrad_float_t *)d_weights, ncol, nrow);

	// compute regularization and initialization for gradients
	conjugrad_float_t *d_reg;
	CHECK_ERR(cudaMalloc((void **) &d_reg, sizeof(conjugrad_float_t)));
	gpu_compute_regularization(d_g_padded, ud->lambda_single, ud->lambda_pair, d_reg, d_histograms, new_nvar, nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD));

	// compute function value
	conjugrad_float_t *d_fx;
	CHECK_ERR(cudaMalloc((void **) &d_fx, sizeof(conjugrad_float_t)));
	gpu_compute_fx((const conjugrad_float_t *)d_precompiled, (const conjugrad_float_t *)d_precompiled_sum, ncol, nrow, d_fx);


	// compute node gradients
	gpu_compute_node_gradients(d_g1_padded, d_precompiled_norm, ncol, nrow);


	// compute edge gradients
	gpu_compute_edge_gradients(d_msa_transposed, d_g2_padded, d_precompiled_norm, ncol, nrow);


	// copy results to CPU
	conjugrad_float_t new_fx, new_reg;
	CHECK_ERR(cudaMemcpy(&new_fx, d_fx, sizeof(conjugrad_float_t), cudaMemcpyDeviceToHost));
	CHECK_ERR(cudaMemcpy(&new_reg, d_reg, sizeof(conjugrad_float_t), cudaMemcpyDeviceToHost));

	conjugrad_float_t fx = new_fx + new_reg;
	return fx;
}


int init_cuda( void *instance ) {

	userdata *ud = (userdata *)instance;
	int ncol = ud->ncol;
	int nrow = ud->nrow;
	int nsingle = ud->nsingle;
	int nvar = ud->nvar;
	int new_nvar = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD) + ncol * ncol * N_ALPHA * N_ALPHA_PAD;

	unsigned char *msa = ud->msa;
	
	// Allocate and copy memory on/to the GPU
	CHECK_ERR(cudaMalloc((void **) &d_msa, sizeof(unsigned char) * ncol * nrow));
	CHECK_ERR(cudaMemcpy(d_msa, msa, sizeof(unsigned char) * ncol * nrow, cudaMemcpyHostToDevice));

	CHECK_ERR(cudaMalloc((void **) &d_precompiled, sizeof(conjugrad_float_t) * ncol * nrow));
	CHECK_ERR(cudaMalloc((void **) &d_precompiled_sum, sizeof(conjugrad_float_t) * ncol * nrow));
	CHECK_ERR(cudaMalloc((void **) &d_precompiled_norm, sizeof(conjugrad_float_t) * ncol * N_ALPHA_PAD * nrow));


	CHECK_ERR(cudaMalloc((void **) &d_histograms, sizeof(conjugrad_float_t) * new_nvar));
	CHECK_ERR(cudaMemset(d_histograms, 0, sizeof(conjugrad_float_t) * new_nvar));

	CHECK_ERR(cudaMalloc((void **) &d_msa_transposed, sizeof(unsigned char) * ncol * nrow));
	gpu_tranpose_msa(d_msa, d_msa_transposed, ncol, nrow);	

	CHECK_ERR(cudaMalloc((void **) &d_weights, sizeof(conjugrad_float_t) * nrow));

	if(ud->reweighting_threshold < 1) {

		CHECK_ERR(cudaMemset(d_weights, 0, sizeof(conjugrad_float_t) * nrow));
		gpu_compute_weights_simple(d_weights, ud->reweighting_threshold, d_msa, nrow, ncol);

		conjugrad_float_t *tmp_weights = (conjugrad_float_t *)malloc(sizeof(conjugrad_float_t) * nrow);
		CHECK_ERR(cudaMemcpy(tmp_weights, d_weights, sizeof(conjugrad_float_t) * nrow, cudaMemcpyDeviceToHost));
		conjugrad_float_t wsum = 0.0;
		conjugrad_float_t wmin = tmp_weights[0], wmax = tmp_weights[0];
		for (int i = 0; i < nrow; i++) {
			conjugrad_float_t wt = tmp_weights[i];
			wsum += wt;
			if(wt > wmax) { wmax = wt; }
			if(wt < wmin) { wmin = wt; }
		}
		printf("Reweighted %d sequences with threshold %.1f to Beff=%g weight mean=%g, min=%g, max=%g\n", nrow, ud->reweighting_threshold, wsum, wsum / nrow, wmin, wmax);
		free(tmp_weights);

	} else {


		conjugrad_float_t *tmp_weights = (conjugrad_float_t *)malloc(sizeof(conjugrad_float_t) * nrow);
		for(int i = 0; i < nrow; i++) {
			tmp_weights[i] = F1;
		}
		CHECK_ERR(cudaMemcpy(d_weights, tmp_weights, sizeof(conjugrad_float_t) * nrow, cudaMemcpyHostToDevice));
		free(tmp_weights);
		printf("Using uniform weights\n");	

	}

	gpu_initialize_histograms_weighted(d_msa, d_histograms, (const conjugrad_float_t *)d_weights, ncol, nrow);

	return EXIT_SUCCESS;
}

int destroy_cuda( void *instance ) {
	CHECK_ERR(cudaFree(d_msa));
	CHECK_ERR(cudaFree(d_msa_transposed));
	CHECK_ERR(cudaFree(d_precompiled));
	CHECK_ERR(cudaFree(d_precompiled_sum));
	CHECK_ERR(cudaFree(d_precompiled_norm));
	CHECK_ERR(cudaFree(d_histograms));
	CHECK_ERR(cudaFree(d_weights));

	return EXIT_SUCCESS;
}
