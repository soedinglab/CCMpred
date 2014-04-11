#include <stdio.h>
#include "arithmetic.h"
#include "ccmpred.h"

void numdif(
	void *instance,
	conjugrad_evaluate_t evaluate,
	conjugrad_float_t *x,
	int b,
	int k,
	int a,
	int j,
	conjugrad_float_t epsilon,
	int n
) {
	userdata *ud = (userdata *)instance;
	int ncol = ud->ncol;
	
	conjugrad_float_t *x_a = conjugrad_malloc(n);
	conjugrad_float_t *x_b = conjugrad_malloc(n);
	conjugrad_float_t *g = conjugrad_malloc(n);
	veccpy(x_a, x, n);
	veccpy(x_b, x, n);


	int i_a = ud->nsingle + x2_index(b,k,a,j);
	int i_b = ud->nsingle + x2_index(a,j,b,k);

	x_a[i_a] -= epsilon;
	x_a[i_b] -= epsilon;
	x_b[i_a] += epsilon;
	x_b[i_b] += epsilon;

	printf("j=%d, k=%d, a=%d, b=%d:\n", j, k, a, b);
	printf("x_a[%6d]=%g, x[%6d]=%g, x_b[%6d]=%g\n", i_a, x_a[i_a], i_a, x[i_a], i_a, x_b[i_a]);
	printf("x_a[%6d]=%g, x[%6d]=%g, x_b[%6d]=%g\n", i_b, x_a[i_b], i_b, x[i_b], i_b, x_b[i_b]);
	
	vec0(g, n);
	conjugrad_float_t f_a = evaluate(instance, x_a, g, n);
	vec0(g, n);
	conjugrad_float_t f_b = evaluate(instance, x_b, g, n);
	vec0(g, n);
	evaluate(instance, x, g, n);

	conjugrad_float_t numdif = (f_b - f_a) / (2.0 * epsilon);

	printf("\t\t\t\t\t g[%6d]=%g\n", i_a, g[i_a]);
	printf("\t\t\t\t\t g[%6d]=%g\n", i_b, g[i_b]);
	printf("\t\t\t\t\tng[%6d]=%g\n", i_a, numdif);
	printf("\t\t\t\t\tng[%6d]=%g\n", i_b, numdif);


	conjugrad_free(x_a);
	conjugrad_free(x_b);
	conjugrad_free(g);

}


