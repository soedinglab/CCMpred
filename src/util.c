#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conjugrad.h"

#include "ccmpred.h"

void die(const char* message) {
	fprintf(stderr, "\nERROR: %s\n\n", message);
	exit(1);
}

void sum_submatrices(conjugrad_float_t *x, conjugrad_float_t *out, int ncol) {

	int nsingle = ncol * (N_ALPHA - 1);
	int nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);
	conjugrad_float_t *x2 = &x[nsingle_padded];

	memset(out, 0, sizeof(conjugrad_float_t) * ncol * ncol);

	conjugrad_float_t xnorm = 0;
	for(int k = 0; k < ncol; k++) {
		for(int j = k+1; j < ncol; j++) {
			for(int a = 0; a < N_ALPHA; a++) {
				for(int b = 0; b < N_ALPHA; b++) {
					conjugrad_float_t w = W(b,k,a,j);
					xnorm += w * w;
				}
			}
		}
	}
	printf("xnorm = %g\n", sqrt(xnorm));

	for(int k = 0; k < ncol; k++) {
		for(int j = 0; j < ncol; j++) {
			conjugrad_float_t mean = 0;
			for(int a = 0; a < N_ALPHA; a++) {
				for(int b = 0; b < N_ALPHA; b++) {
					mean += W(b, k, a, j);
				}
			}

			mean /= (N_ALPHA * N_ALPHA);

			for(int a = 0; a < N_ALPHA - 1; a++) {
				for(int b = 0; b < N_ALPHA - 1; b++) {
					conjugrad_float_t w = W(b,k,a,j) - mean;
					out[k * ncol + j] += w * w;
				}
			}
		}
	}

	for(int k = 0; k < ncol; k++) {
		for(int j = 0; j < ncol; j++) {
			out[k * ncol + j] = sqrt(out[k * ncol + j]);
		}

		out[k * ncol + k] = F0; // 0.0
	}
}

void apc(conjugrad_float_t *mat, int ncol) {

	conjugrad_float_t means[ncol];
	memset(means, 0, sizeof(conjugrad_float_t) * ncol);

	conjugrad_float_t meansum = 0;
	for(int i = 0; i < ncol; i++) {
		for(int j = 0; j < ncol; j++) {
			conjugrad_float_t w = mat[i * ncol + j];
			means[j] += w / ncol;
			meansum += w;
		}
	}
	meansum /= ncol * ncol;


	for(int i = 0; i < ncol; i++) {
		for(int j = 0; j < ncol; j++) {
			mat[i * ncol + j] -= (means[i] * means[j]) / meansum;
		}
	}


	conjugrad_float_t min_wo_diag = 1./0.;
	for(int i = 0; i < ncol; i++) {
		for(int j = i+1; j < ncol; j++) {
			if(mat[i * ncol + j] < min_wo_diag) {
				min_wo_diag = mat[i * ncol + j];
			}
		}
	}

	
	for(int i = 0; i < ncol; i++) {
		for(int j = 0; j < ncol; j++) {
			mat[i * ncol + j] -= min_wo_diag;
		}

		mat[i * ncol + i] = 0;
	}


}

void normalize(conjugrad_float_t *mat, int ncol) {
	conjugrad_float_t min = mat[1];
	conjugrad_float_t max = mat[1];
	for(int i = 0; i < ncol; i++) {
		for(int j = 0; j < ncol; j++) {
			if(i == j) { continue; }

			conjugrad_float_t x = mat[i * ncol + j];
			if(x < min) { min = x; }
			if(x > max) { max = x; }

		}
	}

	conjugrad_float_t range = max - min;
	assert(range != 0);

	for(int i = 0; i < ncol; i++) {
		for(int j = 0; j < ncol; j++) {

			conjugrad_float_t x = mat[i * ncol + j];
			x = (x-min)/range;
			mat[i * ncol + j] = x;

			if(i == j) {
				mat[i * ncol + j] = F0; // 0.0
			}
		}
	}

}
