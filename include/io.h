#include "conjugrad.h"

/**
 * Write out a matrix file to a file descriptor as tab-separated table of floats
 *
 * @param[in] out The file handle to write to
 * @param[in] mat The matrix to output
 * @param[in] ncol The number of columns in the matrix
 * @param[in] nrow The number of rows in the matrix
 */
void write_matrix(FILE *out, conjugrad_float_t *mat, int ncol, int nrow);


/**
 * Write out raw MRF data to a file descriptor as multiple tab-separated tables of floats
 * 
 * @param[in] out The file handle to write to
 * @param[in] x The parameters of the MRF to write out
 * @param[in] ncol The number of columns in the underlying MSA
 */
void write_raw(FILE *out, conjugrad_float_t *x, int ncol);

/**
 * Read initial weights from a file.
 *
 * @param[in] filename The file name to read from
 * @param[in] ud The userdata object to get model information from
 * @param[out] x Memory to write weights to
 */
void read_raw(char *filename, userdata *ud, conjugrad_float_t *x);

#ifdef MSGPACK
void write_raw_msgpack(FILE *out, conjugrad_float_t *x, int ncol, void* meta);
#endif
