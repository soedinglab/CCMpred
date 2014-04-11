/** Sum up amino acid pairing submatrices so we have one score per pair of columns
 * @param[in] x The 21xLx21xL matrix of column and amino acid pairs
 * @param[out] out The LxL matrix of column pairs
 * @param[in] ncol The number of columns in the output matrix (i.e. L)
 */
void sum_submatrices(conjugrad_float_t *x, conjugrad_float_t *out, int ncol);

/** Average product correction
 * @param[in,out] mat The matrix to process
 * @param[in] ncol The number of columns in the matrix
 */
void apc(conjugrad_float_t *mat, int ncol);

/** Linearly re-scale the matrix
 * All matrix elements will be normalized so that the maximum element is 1.0 and the minimum element is 0.0.
 * The diagonal will be ignored when looking at the value range and later set to 0.0
 *
 * @param[in,out] mat The matrix to process
 * @param[in] ncol The number of columns in the matrix
 */
void normalize(conjugrad_float_t *mat, int ncol);


/**
 * Exit the program with an error message.
 *
 * @param[in] message The error message to display
 */
void die(const char* message);
