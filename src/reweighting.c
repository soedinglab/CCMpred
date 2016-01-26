#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "ccmpred.h"
#include "reweighting.h"

void calculate_weights(conjugrad_float_t *w, unsigned char *msa, uint64_t ncol, uint64_t nrow, conjugrad_float_t threshold) {
	int idthres = (int)ceil(threshold * ncol);

	memset(w, 0, sizeof(conjugrad_float_t) * nrow);

	uint64_t nij = nrow * (nrow + 1) / 2;

	#pragma omp parallel
	#pragma omp for nowait
	for(uint64_t ij = 0; ij < nij; ij++) {

		// compute i and j from ij
		// http://stackoverflow.com/a/244550/1181102
		uint64_t i, j;
		{
			uint64_t ii = nrow * (nrow + 1) / 2 - 1 - ij;
			uint64_t K = floor((sqrt(8 * ii + 1) - 1) / 2);
			i = nrow - 1 - K;
			j = ij - nrow * i + i * (i + 1) / 2;
	 	}

		int ids = 0;
		for(uint64_t k = 0; k < ncol; k++) {
			if(msa[msa_index(i, k)] == msa[msa_index(j, k)]) {
				ids++;
			}
		}

		if(ids > idthres) {
			w[i]++;
			w[j]++;
		}
	}

	for(uint64_t i = 0; i < nrow; i++) {
		w[i] = 1./(w[i] - 1);
	}

	conjugrad_float_t wsum = 0;
	conjugrad_float_t wmin = w[0], wmax = w[0];
	for(uint64_t i = 0; i < nrow; i++) {
		conjugrad_float_t wt = w[i];
		wsum += wt;
		if(wt > wmax) { wmax = wt; }
		if(wt < wmin) { wmin = wt; }
	}

	printf("Reweighted %ld sequences with threshold %.1f to Beff=%g weight mean=%g, min=%g, max=%g\n", nrow, threshold, wsum, wsum / nrow, wmin, wmax);

}


void uniform_weights(conjugrad_float_t *w, uint64_t nrow) {
	for(uint64_t i = 0; i < nrow; i++) {
		w[i] = F1;
	}

	printf("Using uniform weights on %ld sequences\n", nrow);
}
