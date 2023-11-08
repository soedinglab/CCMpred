#include <stdbool.h>

#ifndef __LIBCONJ_H__
#define __LIBCONJ_H__

#ifndef CONJUGRAD_FLOAT
#define CONJUGRAD_FLOAT 64
#endif

#if CONJUGRAD_FLOAT == 32
typedef float conjugrad_float_t;

#define F0   0.0f
#define F001 0.01f
#define F02  0.2f
#define F05  0.5f
#define F08  0.8f
#define F1   1.0f
#define F2   2.0f
#define FInf 0.0f/0.0f
#define fsqrt sqrtf
#define flog logf
#define fexp expf
#define fdlog __logf
#define fdexp __expf


#elif CONJUGRAD_FLOAT == 64
typedef double conjugrad_float_t;

#define F0   0.0
#define F001 0.01
#define F02  0.2
#define F05  0.5
#define F08  0.8
#define F1   1.0
#define F2   2.0
#define FInf 0.0/0.0
#define fsqrt sqrt
#define flog log
#define fexp exp
#define fdlog log
#define fdexp exp

#else
#error "Only double-precision (CONJGRAD_FLOAT=64) or single-precision (CONJGRAD_FLOAT=32) supported"
#endif

typedef struct {
	int max_iterations;
	int max_linesearch;
	int k;
	conjugrad_float_t epsilon;	// Tolerance for convergence criterion
	conjugrad_float_t ftol;		// Tolerance for line search sufficient decrease criterion
	conjugrad_float_t wolfe;	// Tolerance for line search curvature criterion
	conjugrad_float_t alpha_mul;	// Line search backtracking factor in range (0,1)
	conjugrad_float_t min_gnorm;    // Value for initial gnorm that will be considered already minimized
} conjugrad_parameter_t;



typedef conjugrad_float_t (*conjugrad_evaluate_t)(
	void *instance,
	const conjugrad_float_t *x,
	conjugrad_float_t *g,
	const int n
);

typedef int (*conjugrad_progress_t)(
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
);

int conjugrad_gpu(
	int n,
	conjugrad_float_t *d_x,
	conjugrad_float_t *d_fx,
	conjugrad_evaluate_t proc_evaluate,
	conjugrad_progress_t proc_progress,
	void *instance,
	conjugrad_parameter_t *param
);

int conjugrad(
	int n,
	conjugrad_float_t *x,
	conjugrad_float_t *fx,
	conjugrad_evaluate_t proc_evaluate,
	conjugrad_progress_t proc_progress,
	void *instance,
	conjugrad_parameter_t *param
);

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
);

conjugrad_parameter_t *conjugrad_init();
conjugrad_float_t *conjugrad_malloc(int n);
void conjugrad_free(conjugrad_float_t *x);


void conjugrad_debug_numdiff(
	int n,
	conjugrad_float_t *x0,
	int i,
	conjugrad_evaluate_t proc_evaluate,
	void *instance,
	conjugrad_float_t epsilon,
	bool have_extra_i,
	int extra_i
);


enum {
	CONJUGRAD_SUCCESS = 0,
	CONJUGRAD_ALREADY_MINIMIZED,
	
	CONJUGRADERR_UNKNOWN = -1024,
	CONJUGRADERR_MAXIMUMLINESEARCH,
	CONJUGRADERR_MAXIMUMITERATION

};


#endif
