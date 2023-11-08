#ifndef CONJUGRAD_ARITHMETIC_H
#define CONJUGRAD_ARITHMETIC_H

#include "conjugrad.h"


// set all vector elements to 0
void vec0(conjugrad_float_t *dst, int n);

// copy src onto dst
void veccpy(conjugrad_float_t *dst, conjugrad_float_t *src, int n);

// dst += src
//void veciaddv(conjugrad_float_t *dst, conjugrad_float_t *src, int n);

// dst *= f
void vecimulc(conjugrad_float_t *dst, conjugrad_float_t f, int n);

// dst -= f * src
void vecifma(conjugrad_float_t *dst, conjugrad_float_t *src, conjugrad_float_t f, int n);

// dst = f * b - a
void vecsfms(conjugrad_float_t *dst, conjugrad_float_t *a, conjugrad_float_t f, conjugrad_float_t *b, int n);

// dst = sum_n v[n] * w[n]
conjugrad_float_t vecdot(conjugrad_float_t *v, conjugrad_float_t *w, int n);

// dst = sum_n v[n]^2
conjugrad_float_t vecnorm(conjugrad_float_t *v, int n);


conjugrad_float_t *vecalloc(int n);

#endif
