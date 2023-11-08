#include <stdbool.h>
#include <stdio.h>
#include "conjugrad.h"


conjugrad_float_t evaluate(
	void *instance,
	const conjugrad_float_t *x,
	conjugrad_float_t *g,
	const int n
) {
	(void)instance;
	(void)n;

	conjugrad_float_t fx = 0.0;


	conjugrad_float_t a = x[0];

	//printf("eval step = %g, a = %g\n", alpha, a);

	conjugrad_float_t *g_a = &g[0];

	*g_a =  0.;

	fx = (a-4)*(a-4); 

	*g_a = 2 * a - 8;

	printf("fx = %g, da = %g\n", fx, g[0]);

	return fx;


}

int progress(
	void *instance,
	const conjugrad_float_t *x,
	const conjugrad_float_t *g,
	const conjugrad_float_t fx,
	const conjugrad_float_t xnorm,
	const conjugrad_float_t gnorm,
	const conjugrad_float_t step,
	int n,
	int k,
	int ls
) {
	(void)instance;
	(void)x;
	(void)g;
	(void)n;

	printf("%d\t%d\t%g\t%g\t%g\t%g\n", k, ls, fx, xnorm, gnorm, step);
	return true;
}

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;

	conjugrad_parameter_t *param = conjugrad_init();
	

	int n = 1;
	conjugrad_float_t *x = conjugrad_malloc(n);
	x[0] = 0;
	conjugrad_float_t fx;

	int ret = conjugrad(n, x, &fx, evaluate, progress, NULL, param);

	printf("Return code %d\n", ret);

}
