#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ccmpred.h"
#include "io.h"
#include "conjugrad.h"

void write_matrix(FILE *out, conjugrad_float_t *mat, int ncol, int nrow) {
	for(int i = 0; i < nrow; i++) {
		for(int j = 0; j < ncol; j++) {
			fprintf(out, "%.20e\t", mat[i * ncol + j]);
		}

		fprintf(out, "\n");
	}
}


void write_raw(FILE *out, conjugrad_float_t *x, int ncol) {

	int nsingle = ncol * (N_ALPHA - 1);
	int nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);

	conjugrad_float_t *x1 = x;
	//conjugrad_float_t *x2 = &x[nsingle];
	conjugrad_float_t *x2 = &x[nsingle_padded];
	(void)x2;

	for(int i = 0; i < ncol; i++) {
		for(int a = 0; a < N_ALPHA - 1; a++) {
			fprintf(out, "%g\t", V(i, a));
		}
		fprintf(out, "\n");
	}


	for(int i = 0; i < ncol; i++) {
		for(int j = i+1; j < ncol; j++) {
			fprintf(out, "# %d %d\n", i, j);
			for(int a = 0; a < N_ALPHA; a++) {
				for(int b = 0; b < N_ALPHA; b++) {
					fprintf(out, "%.20e\t", W(b,j,a,i));
				}
				fprintf(out, "\n");
			}
		}
	}
}

void read_raw(char *filename, userdata *ud, conjugrad_float_t *x) {
	FILE *f = fopen(filename, "r");
	char *line = (char*) malloc(sizeof(char) * 8192);
	char* expected = malloc(1024);
	int ncol = ud->ncol;

	conjugrad_float_t *x1 = x;
	conjugrad_float_t *x2 = x + ud->nsingle;

	for(int i = 0; i < ncol; i++) {
		fgets(line, 8192, f);
		char *token = line;
		for(int a = 0; a < N_ALPHA-1; a++) {
			V(i, a) = strtod(token, &token);
		}
	}

	for(int i = 0; i < ncol; i++) {
		for(int j = i+1; j < ncol; j++) {
			fgets(line, 8192, f);

			snprintf(expected, 1024, "# %d %d\n", i, j);
			assert(strcmp(line, expected) == 0);

			for(int a = 0; a < N_ALPHA; a++) {
				fgets(line, 8192, f);
				char *token = line;
				for(int b = 0; b < N_ALPHA; b++) {
					W(b, j, a, i) = W(a, i, b, j) = strtod(token, &token);
				}
			}
		}


	}

	free(expected);
	free(line);
	fclose(f);
}

