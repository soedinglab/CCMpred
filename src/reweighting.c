#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "ccmpred.h"
#include "reweighting.h"

void calculate_weights(conjugrad_float_t *w, unsigned char *msa, int ncol, int nrow, conjugrad_float_t threshold) {
	int idthres = (int)ceil(threshold * ncol);

	memset(w, 0, sizeof(conjugrad_float_t) * nrow);

	for(int i = 0; i < nrow; i++) {
		for(int j = i+1; j < nrow; j++) {

			int ids = 0;
			for(int k = 0; k < ncol; k++) {
				if(msa[msa_index(i, k)] == msa[msa_index(j, k)]) {
					ids++;
				}
			}

			if(ids > idthres) {
				w[i]++;
				w[j]++;
			}
		}
	}

	for(int i = 0; i < nrow; i++) {
		w[i] = 1./(1 + w[i]);
	}

	conjugrad_float_t wsum = 0;
	conjugrad_float_t wmin = w[0], wmax = w[0];
	for(int i = 0; i < nrow; i++) {
		conjugrad_float_t wt = w[i];
		wsum += wt;
		if(wt > wmax) { wmax = wt; }
		if(wt < wmin) { wmin = wt; }
	}

	printf("Reweighted %d sequences with threshold %.1f to Beff=%g weight mean=%g, min=%g, max=%g\n", nrow, threshold, wsum, wsum / nrow, wmin, wmax);

}


void uniform_weights(conjugrad_float_t *w, int nrow) {
	for(int i = 0; i < nrow; i++) {
		w[i] = F1;
	}

	printf("Using uniform weights on %d sequences\n", nrow);
}
