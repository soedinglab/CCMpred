#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include "arithmetic.h"
#include "conjugrad.h"

void vec0(conjugrad_float_t *dst, int n) {
	memset(dst, 0, sizeof(conjugrad_float_t) * n);
}

void veccpy(conjugrad_float_t *dst, conjugrad_float_t *src, int n) {
	memcpy(dst, src, sizeof(conjugrad_float_t) * n);
}

void vecimulc(conjugrad_float_t *dst, conjugrad_float_t f, int n) {
	for(int i = 0; i < n; i++) {
		dst[i] *= f;
	}
}

void vecifma(conjugrad_float_t *dst, conjugrad_float_t *src, conjugrad_float_t f, int n) {
	for(int i = 0; i < n; i++) {
		dst[i] += f * src[i];
	}
}

void vecsfms(conjugrad_float_t *dst, conjugrad_float_t *a, conjugrad_float_t f, conjugrad_float_t *b, int n) {
	for(int i = 0; i < n; i++) {
		dst[i] = f * b[i] - a[i];
	}
}

conjugrad_float_t vecnorm(conjugrad_float_t *v, int n) {
	conjugrad_float_t sum = 0.;
	for(int i = 0; i < n; i++) {
		sum += v[i] * v[i];
	}
	return sum;
}

conjugrad_float_t vecdot(conjugrad_float_t *v, conjugrad_float_t *w, int n) {
	conjugrad_float_t sum = 0.;
	for(int i = 0; i < n; i++) {
		sum += v[i] * w[i];
	}
	return sum;
}

conjugrad_float_t *vecalloc(int n) {
	return (conjugrad_float_t *) malloc(sizeof(conjugrad_float_t) * n);
}
