#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ccmpred.h"
#include "io.h"
#include "conjugrad.h"

#ifdef MSGPACK
#include <msgpack.h>

#ifdef JANSSON
#include <jansson.h>
#include "meta.h"
#endif

#endif

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

#ifdef MSGPACK
void write_raw_msgpack(FILE *out, conjugrad_float_t *x, int ncol, void *meta) {
	int nsingle = ncol * (N_ALPHA - 1);
	int nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);

	conjugrad_float_t *x1 = x;
	conjugrad_float_t *x2 = &x[nsingle_padded];
	(void)x2;

	msgpack_sbuffer* buffer = msgpack_sbuffer_new();
	msgpack_packer* pk = msgpack_packer_new(buffer, msgpack_sbuffer_write);

#ifdef JANSSON
	if(meta != NULL) {
		msgpack_pack_map(pk, 5);
		meta_write_msgpack(pk, (json_t *)meta);
	} else {
		msgpack_pack_map(pk, 4);
	}
#else
	msgpack_pack_map(pk, 4);
#endif

	msgpack_pack_str(pk, 6);
	msgpack_pack_str_body(pk, "format", 6);
	msgpack_pack_str(pk, 5);
	msgpack_pack_str_body(pk, "ccm-1", 5);

	msgpack_pack_str(pk, 4);
	msgpack_pack_str_body(pk, "ncol", 4);
	msgpack_pack_int32(pk, ncol);

	msgpack_pack_str(pk, 8);
	msgpack_pack_str_body(pk, "x_single", 8);
	msgpack_pack_array(pk, ncol * (N_ALPHA - 1));
	for(int i = 0; i < ncol; i++) {
		for(int a = 0; a < N_ALPHA - 1; a++) {
			#if CONJUGRAD_FLOAT == 32
				msgpack_pack_float(pk, V(i, a));
			#elif CONJUGRAD_FLOAT == 64
				msgpack_pack_double(pk, V(i, a));
			#endif
		}
	}

	msgpack_pack_str(pk, 6);
	msgpack_pack_str_body(pk, "x_pair", 6);

	int nedge = ncol * (ncol - 1) / 2;
	msgpack_pack_map(pk, nedge);

	char sbuf[8192];
	for(int i = 0; i < ncol; i++) {
		for(int j = i + 1; j < ncol; j++) {

			int nchar = snprintf(sbuf, 8192, "%d/%d", i, j);

			msgpack_pack_str(pk, nchar);
			msgpack_pack_str_body(pk, sbuf, nchar);

			msgpack_pack_map(pk, 3);
			
			msgpack_pack_str(pk, 1);
			msgpack_pack_str_body(pk, "i", 1);
			msgpack_pack_int32(pk, i);

			msgpack_pack_str(pk, 1);
			msgpack_pack_str_body(pk, "j", 1);
			msgpack_pack_int32(pk, j);

			msgpack_pack_str(pk, 1);
			msgpack_pack_str_body(pk, "x", 1);

			msgpack_pack_array(pk, N_ALPHA * N_ALPHA);
			for(int a = 0; a < N_ALPHA; a++) {
				for(int b = 0; b < N_ALPHA; b++) {
					#if CONJUGRAD_FLOAT == 32
						msgpack_pack_float(pk, W(b, j, a, i));
					#elif CONJUGRAD_FLOAT == 64
						msgpack_pack_double(pk, W(b, j, a, i));
					#endif
				}
			}
		}
	}


	fwrite(buffer->data, buffer->size, 1, out);

	msgpack_sbuffer_free(buffer);
	msgpack_packer_free(pk);
}
#endif
