#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "arithmetic.h"
#include "conjugrad.h"

#include <immintrin.h>

#ifndef CONJUGRAD_SIMD
#define CONJUGRAD_SIMD 1
#endif

#if CONJUGRAD_FLOAT == 32
typedef float fval;

#if CONJUGRAD_SIMD == 1 // SSE
#define STRIDE 4
typedef __m128 simd;
#define simdset1 _mm_set1_ps
#define simdload _mm_load_ps
#define simdmul _mm_mul_ps
#define simdadd _mm_add_ps
#define simdsub _mm_sub_ps
#define simdstore _mm_store_ps

#elif CONJUGRAD_SIMD == 2 // AVX
#define STRIDE 8
typedef __m256 simd;
#define simdset1 _mm256_set1_ps
#define simdload _mm256_load_ps
#define simdmul _mm256_mul_ps
#define simdadd _mm256_add_ps
#define simdsub _mm256_sub_ps
#define simdstore _mm256_store_ps
#endif // CONJUGRAD_SIMD

#else // CONJUGRAD_FLOAT
typedef double fval;

#if CONJUGRAD_SIMD == 1 // SSE
#define STRIDE 2
typedef __m128d simd;
#define simdset1 _mm_set1_pd
#define simdload _mm_load_pd
#define simdmul _mm_mul_pd
#define simdadd _mm_add_pd
#define simdsub _mm_sub_pd
#define simdstore _mm_store_pd

#elif CONJUGRAD_SIMD == 2 // AVX
#define STRIDE 4
typedef __m256d simd;
#define simdset1 _mm256_set1_pd
#define simdload _mm256_load_pd
#define simdmul _mm256_mul_pd
#define simdadd _mm256_add_pd
#define simdsub _mm256_sub_pd
#define simdstore _mm256_store_pd
#endif // CONJUGRAD_SIMD
#endif // CONJUGRAD_FLOAT

void vec0(conjugrad_float_t *dst, int n) {
	memset(dst, 0, sizeof(conjugrad_float_t) * n);
}

void veccpy(conjugrad_float_t *dst, conjugrad_float_t *src, int n) {
	memcpy(dst, src, sizeof(conjugrad_float_t) * n);
}

void vecimulc(conjugrad_float_t *dst, conjugrad_float_t f, int n) {
	simd _dst;
	simd _f = simdset1(f);

	fval *pdst = (fval *)dst;

	int m = (n / STRIDE) * STRIDE;
	for(int i = 0; i < m; i+= STRIDE) {
		_dst = simdload(pdst);
		_dst = simdmul(_dst, _f);
		simdstore(pdst, _dst);

		pdst += STRIDE;
	}

	for(int i = m; i < n; i++) {
		dst[i] *= f;
	}
}

void vecifma(conjugrad_float_t *dst, conjugrad_float_t *src, conjugrad_float_t f, int n) {
	simd _dst;
	simd _src;

	simd _f = simdset1(f);

	fval *pdst = (fval *)dst;
	fval *psrc = (fval *)src;

	int m = (n / STRIDE) * STRIDE;
	for(int i = 0; i < m; i += STRIDE) {
		_dst = simdload(pdst);
		_src = simdload(psrc);
		_src = simdmul(_src, _f);
		_dst = simdadd(_dst, _src);
		simdstore(pdst, _dst);

		pdst += STRIDE;
		psrc += STRIDE;
	}

	for(int i = m; i < n; i++) {
		dst[i] += f * src[i];
	}
}

void vecsfms(conjugrad_float_t *dst, conjugrad_float_t *a, conjugrad_float_t f, conjugrad_float_t *b, int n) {
	simd _dst;
	simd _a;
	simd _b;

	simd _f = simdset1(f);

	fval *pdst = (fval *)dst;
	fval *pa = (fval *)a;
	fval *pb = (fval *)b;

	int m = (n / STRIDE) * STRIDE;
	for(int i = 0; i < m; i += STRIDE) {
		_a = simdload(pa);
		_b = simdload(pb);
		_dst = simdmul(_b, _f);
		_dst = simdsub(_dst, _a);
		simdstore(pdst, _dst);

		pdst += STRIDE;
		pa += STRIDE;
		pb += STRIDE;
	}

	for(int i = m; i < n; i++) {
		dst[i] = a[i] + f * b[i];
	}

}

conjugrad_float_t vecnorm(conjugrad_float_t *v, int n) {

	simd _sum = simdset1(.0f);	
	simd _v;


	fval *pv = (fval *)v;
	int m = (n / STRIDE) * STRIDE;
	for(int i = 0; i < m; i += STRIDE) {
		_v = simdload(pv);
		_v = simdmul(_v, _v);
		_sum = simdadd(_sum, _v);
		pv += STRIDE;
	}

	fval psum[STRIDE];
	simdstore(psum, _sum);

	fval sum = .0f;
	for(int i = 0; i < STRIDE; i++) {
		sum += psum[i];
	}

	for(int i = m; i < n; i++) {
		sum += v[i] * v[i];
	}

	return sum;
}

conjugrad_float_t vecdot(conjugrad_float_t *v, conjugrad_float_t *w, int n) {
	simd _sum = simdset1(.0f);	
	simd _v;
	simd _w;

	fval *pv = (fval *)v;
	fval *pw = (fval *)w;
	int m = (n / STRIDE) * STRIDE;
	for(int i = 0; i < m; i += STRIDE) {
		_v = simdload(pv);
		_w = simdload(pw);
		_v = simdmul(_v, _w);
		_sum = simdadd(_sum, _v);
		pv += STRIDE;
		pw += STRIDE;
	}

	fval psum[STRIDE];
	simdstore(psum, _sum);

	fval sum = .0f;
	for(int i = 0; i < STRIDE; i++) {
		sum += psum[i];
	}
	for(int i = m; i < n; i++) {
		sum += v[i] * w[i];
	}

	return sum;
}

conjugrad_float_t *vecalloc(int n) {
	conjugrad_float_t *out;
	posix_memalign((void **)&out, sizeof(conjugrad_float_t) * STRIDE, sizeof(conjugrad_float_t) * n);
	return out;
}
