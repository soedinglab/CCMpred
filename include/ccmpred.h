#include "conjugrad.h"

#define N_ALPHA 21

#ifdef PADDING
#define N_ALPHA_PAD 32
#else
#define N_ALPHA_PAD 21
#endif

#define x1_index(j,a) (a) * (ncol) + j
#define V(j,a) x1[x1_index(j,a)]
#define G1(j,a) g1[x1_index(j,a)]
#define L1(j,a) l1[x1_index(j,a)]

#define x2_index(b,k,a,j) (((b) * ncol + (k)) * (N_ALPHA_PAD) + (a)) * ncol + j
#define W(b,k,a,j) x2[x2_index(b,k,a,j)]
#define G2(b,k,a,j) g2[x2_index(b,k,a,j)]
#define L2(b,k,a,j) l2[x2_index(b,k,a,j)]

#define msa_index(i,j) (i) * ncol + j
#define X(i,j) ud->msa[msa_index(i,j)]

#define pc_index(a,s) (a) * ncol + (s)
#define PC(a,s) precomp[pc_index(a,s)]
#define PCN(a,s) precomp_norm[pc_index(a,s)]

#define OMP_pc_index(i,a,s) ((i) * N_ALPHA + (a)) * ncol + (s)
#define OMP_PC(i,a,s) precomp[OMP_pc_index(i,a,s)]
#define OMP_PCN(i,a,s) precomp_norm[OMP_pc_index(i,a,s)]


#ifndef __VERSION
#define __VERSION "unknown"
#endif

/** User data passed to the LBFGS optimizer which will be available in evaluate calls
 */
typedef struct userdata {

	/** The MSA to learn
	 */
	unsigned char* msa;

	/** The sequence weights
	 */
	conjugrad_float_t *weights;

	/** The number of columns in the MSA (i.e. L)
	 */
	int ncol;

	/** The number of rows in the MSA (i.e. N)
	 */
	int nrow;

	/**
	 * The number of single emission potentials
	 */
	int nsingle;

	/**
	 * The number of variables
	 */
	int nvar;

	/**
	 * Extra data for the actual evaluate implementation
	 */
	void *extra;

	/**
	 * Single emission regularization coefficients
	 */
	conjugrad_float_t lambda_single;

	/**
	 * Pairwise emisssion regularization coefficients
	 */
	conjugrad_float_t lambda_pair;

	/**
	 * Threshold for reweighting
	 */
	conjugrad_float_t reweighting_threshold;


	/**
	 * Pointer to metadata step array
	 */
	void *meta_steps;
} userdata;

