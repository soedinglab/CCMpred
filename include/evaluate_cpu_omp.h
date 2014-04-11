#define G2L(b,k,a,j) g2l[x2_index(b,k,a,j)]
#include "conjugrad.h"

typedef struct omp_extra_userdata {

	/**
	 * Extra memory for symmetric access
	 */
	conjugrad_float_t *g2;

} omp_extra_userdata;


/** Callback for LBFGS optimization to calculate function value and gradient
 * @param[in] instance The user data passed to the LBFGS optimizer
 * @param[in] x The current variable assignments
 * @param[out] g The current gradient
 * @param[in] nvar The number of variables
 * @param[in] step The step size for the current iteration
 */
conjugrad_float_t evaluate_cpu_omp (
	void *instance,
	const conjugrad_float_t *x,
	conjugrad_float_t *g,
	const int nvar
);


int init_cpu_omp( void *instance );
int destroy_cpu_omp( void *instance );
