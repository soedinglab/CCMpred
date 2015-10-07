#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include "conjugrad.h"

#include "sequence.h"
#include "ccmpred.h"
#include "evaluate_cpu_omp.h"
#include "util.h"
#include "reweighting.h"

#include <omp.h>

conjugrad_float_t evaluate_cpu_omp(
	void *instance,
	const conjugrad_float_t *x,
	conjugrad_float_t *g,
	const int nvar_padded
) {

	userdata *ud = (userdata *)instance;
	omp_extra_userdata *udx = (omp_extra_userdata *)ud->extra;

	int ncol = ud->ncol;
	int nrow = ud->nrow;
	int nsingle = ud->nsingle;
	int nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);

	conjugrad_float_t lambda_single = ud->lambda_single;
	conjugrad_float_t lambda_pair = ud->lambda_pair;

	const conjugrad_float_t *x1 = x;
	const conjugrad_float_t *x2 = &x[nsingle_padded];

	conjugrad_float_t *g1 = g;
	conjugrad_float_t *g2l = &g[nsingle_padded];

	conjugrad_float_t *g2 = udx->g2;

	// set fx and gradient to 0 initially
	conjugrad_float_t fx = F0; // 0.0

	memset(g, 0, sizeof(conjugrad_float_t) * nvar_padded);
	memset(g2, 0, sizeof(conjugrad_float_t) * (nvar_padded - nsingle_padded));

	conjugrad_float_t *precomp      = (conjugrad_float_t *)malloc(sizeof(conjugrad_float_t) * nrow * N_ALPHA_PAD * ncol);
	conjugrad_float_t *precomp_sum  = (conjugrad_float_t *)malloc(sizeof(conjugrad_float_t) * nrow * ncol);
	conjugrad_float_t *precomp_norm = (conjugrad_float_t *)malloc(sizeof(conjugrad_float_t) * nrow * N_ALPHA_PAD * ncol);
	memset(precomp_sum, 0, sizeof(conjugrad_float_t) * nrow * ncol);

	//omp_set_num_threads(6);

	#pragma omp parallel for
	for(int i = 0; i < nrow; i++) {

		// compute PC(a,s) = V_s(a) + sum(k \in V_s) w_{sk}(a, X^i_k)
		for(int a = 0; a < N_ALPHA-1; a++) {
			for(int s = 0; s < ncol; s++) {
				OMP_PC(i,a,s) = V(s,a);
			}
		}
		
		for(int s = 0; s < ncol; s++) {
			OMP_PC(i,N_ALPHA - 1, s) = 0;
		}


		for(int k = 0; k < ncol; k++) {
			unsigned char xik = X(i,k);

			conjugrad_float_t *pp = &OMP_PC(i, 0, 0);
			const conjugrad_float_t *pw = &W(xik, k, 0, 0);

			for(int a = 0; a < N_ALPHA * ncol; a++) {
				*pp++ += *pw++;
			}
		}

		// compute precomp_sum(s) = log( sum(a=1..21) exp(PC(a,s)) )
		for(int a = 0; a < N_ALPHA; a++) {
			for(int s = 0; s < ncol; s++) {
				precomp_sum[i * ncol + s] += fexp(OMP_PC(i,a,s));
			}
		}

		for(int s = 0; s < ncol; s++) {
			precomp_sum[i * ncol + s] = flog(precomp_sum[i * ncol + s]);
		}

		for(int a = 0; a < N_ALPHA; a++) {
			for(int s = 0; s < ncol; s++) {
				OMP_PCN(i,a,s) = fexp( OMP_PC(i,a, s) - precomp_sum[i * ncol + s] );
			}
		}
	} // i

	for (int i = 0; i < nrow; i++) {
		conjugrad_float_t weight = ud->weights[i];
		for (int k = 0; k < ncol; k++) {
			unsigned char xik = X(i,k);
			fx += weight * (precomp_sum[i * ncol + k] - OMP_PC(i,xik,k));
		}
	}

	#pragma omp parallel for
	for (int k = 0; k < ncol; k++) {
		for (int i = 0; i < nrow; i++) {
			conjugrad_float_t weight = ud->weights[i];
			unsigned char xik = X(i,k);
			if(xik < N_ALPHA - 1) {
				G1(k, xik) -= weight;
			}
			for (int a = 0; a < N_ALPHA - 1; a++) {
				G1(k,a) += weight * OMP_PCN(i,a,k);
			}
		}
	}


	for(int i = 0; i < nrow; i++) {
		conjugrad_float_t weight = ud->weights[i];

		#pragma omp parallel for
		for(int k = 0; k < ncol; k++) {

			unsigned char xik = X(i,k);

			for(int j = 0; j < ncol; j++) {
				int xij = X(i,j);
				G2(xik, k, xij, j) -= weight;
			}

			conjugrad_float_t *pg = &G2(xik, k, 0, 0);
			conjugrad_float_t *pp = &OMP_PCN(i, 0, 0);

			for(int a = 0; a < N_ALPHA * ncol; a++) {
				*pg++ += weight * *pp++;
			}
		}
	} // i


	// add transposed onto un-transposed
	for(int b = 0; b < N_ALPHA; b++) {
		for(int k = 0; k < ncol; k++) {
			for(int a = 0; a < N_ALPHA; a++) {
				for(int j = 0; j < ncol; j++) {
					G2L(b, k, a, j) = G2(b, k, a, j) + G2(a, j, b, k);
				}
			}
		}
	}

	// set gradients to zero for self-edges
	for(int b = 0; b < N_ALPHA; b++) {
		for(int k = 0; k < ncol; k++) {
			for(int a = 0; a < N_ALPHA; a++) {
				G2L(b, k, a, k) = 0;
			}
		}
	}

	// regularization
	conjugrad_float_t reg = F0; // 0.0

	#pragma omp parallel for ordered reduction (+:reg)
	for(int v = 0; v < nsingle; v++) {
		reg += lambda_single * x[v] * x[v];
		g[v] += F2 * lambda_single * x[v]; // F2 is 2.0
	}

	for(int v = nsingle_padded; v < nvar_padded; v++) {
		reg += F05 * lambda_pair * x[v] * x[v];
		g[v] += F2 * lambda_pair * x[v]; // F2 is 2.0
	}

	fx += reg;

	free(precomp);
	free(precomp_sum);
	free(precomp_norm);


	return fx;
}

int init_cpu_omp(void* instance) {


	userdata *ud = (userdata *)instance;
	omp_extra_userdata *udx = (omp_extra_userdata *)malloc(sizeof(omp_extra_userdata));
	ud->extra = udx;

	int ncol = ud->ncol;

	udx->g2 = conjugrad_malloc(ncol * ncol * N_ALPHA * N_ALPHA_PAD);
	if(udx->g2 == NULL) {
		die("ERROR: Not enough memory to allocate temp g2 matrix!");
	}

	if(ud->reweighting_threshold != F1) {
		calculate_weights(ud->weights, ud->msa, ncol, ud->nrow, ud->reweighting_threshold);
	} else {
		uniform_weights(ud->weights, ud->nrow);
	}

	return true;

}

int destroy_cpu_omp(void* instance) {

	userdata *ud = (userdata *)instance;
	omp_extra_userdata *udx = (omp_extra_userdata *)ud->extra;

	conjugrad_free(udx->g2);
	return EXIT_SUCCESS;
}
