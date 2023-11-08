#include "conjugrad.h"
#include "arithmetic.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


void swap(conjugrad_float_t **a, conjugrad_float_t **b) {
	conjugrad_float_t *c = *a;
	*a = *b;
	*b = c;
}

int conjugrad(
	int n,
	conjugrad_float_t *x,
	conjugrad_float_t *fx,
	conjugrad_evaluate_t proc_evaluate,
	conjugrad_progress_t proc_progress,
	void *instance,
	conjugrad_parameter_t *param
) {

	conjugrad_float_t alpha, alphaprev, dg, dgprev, beta, gprevnorm, gnorm, xnorm;
	conjugrad_float_t *g = conjugrad_malloc(n);
	conjugrad_float_t *s = conjugrad_malloc(n);
	int n_linesearch, n_iter = 0;
	int ret = CONJUGRADERR_UNKNOWN;
	conjugrad_float_t *k_last_fx = (conjugrad_float_t *)malloc(sizeof(conjugrad_float_t) * param->k);
	conjugrad_float_t *check_fx = k_last_fx;

	vec0(s, n);

	*fx = proc_evaluate(instance, x, g, n);

	gnorm = vecnorm(g, n);
	xnorm = vecnorm(x, n);

	if(gnorm <= param->min_gnorm || gnorm / xnorm <= param->epsilon) {
		ret = CONJUGRAD_ALREADY_MINIMIZED;
		goto conjugrad_exit;
	}

	alpha = F1/fsqrt(gnorm);

	while(true) {
		if(n_iter >= param->max_iterations) { 
			ret = CONJUGRADERR_MAXIMUMITERATION;
			break;
		};

		// \delta x_n = - g_n
		//vecimulc(g, -1, n);

		if(n_iter > 0) {
			// fletcher-reeves: beta_n = ||x_n|| / ||x_{n-1}||
			beta = gnorm / gprevnorm;

			// s_n = \beta_n * s_{n-1} - g_n
			vecsfms(s, g, beta, s, n);
			dg = vecdot(s, g, n);
			alpha = alphaprev * dgprev / dg;

		} else {
			// s_0 = -g_0
			veccpy(s, g, n);
			vecimulc(s, -1, n);
			dg = vecdot(s, g, n);
		}

		n_linesearch = linesearch(n, x, fx, g, s, &alpha, proc_evaluate, instance, param);

		gprevnorm = gnorm;
		gnorm = vecnorm(g, n);
		xnorm = vecnorm(x, n);
		alphaprev = alpha;
		dgprev = dg;

		if(n_linesearch < 0) {
			ret = n_linesearch;
			break;
		}

		int pos = n_iter % param->k;
		check_fx = k_last_fx + pos;

		if (n_iter >= param->k) {
			conjugrad_float_t rel_change = (*check_fx - *fx) / *check_fx;
			if (rel_change < param->epsilon) {
				ret = CONJUGRAD_SUCCESS;
				break;
			}
		}

		*check_fx = *fx;

		n_iter++;
		proc_progress(instance, x, g, *fx, xnorm, gnorm, alpha, n, n_iter, n_linesearch);

		// convergence check
		//if(xnorm < 1.0) { xnorm = 1.0; }
		//if(gnorm / xnorm <= param->epsilon) {
		//	ret = CONJUGRAD_SUCCESS;
		//	break;
		//}
	}

	conjugrad_exit:	

	conjugrad_free(g);
	conjugrad_free(s);
	free(k_last_fx);
	return ret;
}


conjugrad_parameter_t *conjugrad_init() {
	conjugrad_parameter_t *out = (conjugrad_parameter_t *)malloc(sizeof(conjugrad_parameter_t));
	out->max_linesearch = 100;
	out->max_iterations = 1000;
	out->epsilon = 1e-5;
	out->ftol = 1e-4;
	out->wolfe = 0.1;
	out->alpha_mul = 0.5;
	out->min_gnorm = 1e-8;

	return out;
}

int linesearch(
	int n,
	conjugrad_float_t *x,
	conjugrad_float_t *fx,
	conjugrad_float_t *g,
	conjugrad_float_t *s,
	conjugrad_float_t *alpha,
	conjugrad_evaluate_t proc_evaluate,
	void *instance,
	conjugrad_parameter_t *param
) {

	conjugrad_float_t fx_step;
	int n_linesearch = 0;

	conjugrad_float_t dginit = vecdot(g, s, n);
	conjugrad_float_t dgtest = dginit * param->ftol;
	conjugrad_float_t dg;
	conjugrad_float_t finit = *fx;
	conjugrad_float_t prevalpha = 0;

	while(true) {
		if(n_linesearch >= param->max_linesearch) { return CONJUGRADERR_MAXIMUMLINESEARCH; }
		n_linesearch++;

		// do step
		vecifma(x, s, *alpha - prevalpha, n);

		// x_{n+1} is available implicitly by x_{n+1} = x_n + alpha * s
		fx_step = proc_evaluate(instance, x, g, n);

		// armijo condition
		if(fx_step <= finit + *alpha * dgtest) {

			// wolfe condition (curvature)
			dg = vecdot(s, g, n);
			if(dg < param->wolfe * dginit) {
				*fx = fx_step;
				return n_linesearch;
			}


		}

		prevalpha = *alpha;
		*alpha *= param->alpha_mul;
	}
}



conjugrad_float_t *conjugrad_malloc(int n) {
	return vecalloc(n); 
}

void conjugrad_free(conjugrad_float_t *x) {
	free(x);
}
