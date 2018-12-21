#ifndef __FAST_MATRIX_OPERATION_H__
#define __FAST_MATRIX_OPERATION_H__

#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

namespace mini_cnn
{
	static inline nn_float vec_dot(const nn_float *nn_restrict v1, const nn_float *nn_restrict v2, nn_int len)
	{
		nn_float s = 0;
		for (nn_int i = 0; i < len; ++i)
		{
			s += v1[i] * v2[i];
		}
		return s;
	}

	// m: = m1 * m2
	//
	// gemm (general matrix multiply matrix) 
	// m1: h1 X w1
	// m2: h2 X w2
	static inline void gemm(nn_float *m1, nn_int w1, nn_int h1
		, nn_float *m2, nn_int w2, nn_int h2
		, nn_float *m, nn_int w, nn_int h)
	{
		nn_assert(w1 == h2);
		nn_assert(h1 == h && w2 == w);
		for (nn_int i = 0; i < h; ++i)
		{
			for (nn_int j = 0; j < w; ++j)
			{
				nn_float dot = 0;
				nn_float *pa = &m1[i * w1];
				for (nn_int k = 0; k < w1; ++k)
				{
					dot += pa[k] * m2[k * w2 + j];
				}
				m[i * w + j] = dot;
			}
		}
	}

	// z := m * x + y
	//
	// matrix multiply vector
	// m: matrix
	// x, y, z: vector
	static inline void fo_mvv_v(nn_float *nn_restrict m, nn_int w, nn_int h
		, nn_float *nn_restrict x
		, nn_float *nn_restrict y
		, nn_float *nn_restrict z)
	{
		for (nn_int i = 0; i < h; ++i)
		{
			nn_float dot = vec_dot(&m[i * w], &x[0], w);
			z[i] = dot + y[i];
		}
	}

	// m := x * y
	// 
	// get matrix by vector multiply vector
	// m: matrix with shape of h X w
	// x, y: vector
	static inline void fo_vv_m(nn_float *nn_restrict x, nn_int h, nn_float *nn_restrict y, nn_int w, nn_float *nn_restrict m)
	{
		for (nn_int i = 0; i < h; ++i)
		{
			nn_float *nn_restrict vec_m = &m[i * w];
			nn_float xi = x[i];
			for (nn_int j = 0; j < w; ++j)
			{
				vec_m[j] = xi * y[j];
			}
		}
	}

	// y := m * x
	// 
	// get vector by matrix multiply vector
	// m: matrix with shape of h X w
	// x, y: vector
	static inline void fo_mv_v(const nn_float *nn_restrict m, nn_int h, nn_int w, const nn_float *nn_restrict x, nn_float *nn_restrict y)
	{
		for (nn_int i = 0; i < h; ++i)
		{
			const nn_float *nn_restrict vec_i = &m[i * w];
			y[i] = vec_dot(vec_i, x, w);
		}
	}

	static inline void fo_mv_v_accum(const nn_float *nn_restrict m, nn_int h, nn_int w, const nn_float *nn_restrict x, nn_float *nn_restrict y)
	{
		for (nn_int i = 0; i < h; ++i)
		{
			const nn_float *nn_restrict vec_i = &m[i * w];
			y[i] += vec_dot(vec_i, x, w);
		}
	}

}
#endif //__FAST_MATRIX_OPERATION_H__
