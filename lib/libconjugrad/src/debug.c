#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#include "conjugrad.h"

void conjugrad_debug_numdiff(
	int n,
	conjugrad_float_t *x0,
	int i,
	conjugrad_evaluate_t proc_evaluate,
	void *instance,
	conjugrad_float_t epsilon,
	bool have_extra_i,
	int extra_i
) {

	conjugrad_float_t *xA = conjugrad_malloc(n);
	conjugrad_float_t *xB = conjugrad_malloc(n);
	conjugrad_float_t *g = conjugrad_malloc(n);

	memcpy(xA, x0, sizeof(conjugrad_float_t) * n);
	memcpy(xB, x0, sizeof(conjugrad_float_t) * n);

	xA[i] -= epsilon;
	xB[i] += epsilon;

	if(have_extra_i) {
		xA[extra_i] -= epsilon;
		xB[extra_i] += epsilon;
	}

	conjugrad_float_t fxA = proc_evaluate(instance, xA, g, n);	
	conjugrad_float_t fxB = proc_evaluate(instance, xB, g, n);	
	conjugrad_float_t fx = proc_evaluate(instance, x0, g, n);	

	conjugrad_float_t gNumeric = (fxB - fxA) / (2 * epsilon);
	conjugrad_float_t gSymbolic = g[i];
	
	printf("\t@%8d: gNum = %15g gSym = %15g", i, gNumeric, gSymbolic);

	if(have_extra_i) {
		printf("\t@%8d: gSym = %15g", extra_i, g[extra_i]);

	}

	printf("\n");

	conjugrad_free(xA);
	conjugrad_free(xB);
	conjugrad_free(g);
}
