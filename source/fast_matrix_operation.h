#ifndef __FAST_MATRIX_OPERATION_H__
#define __FAST_MATRIX_OPERATION_H__

#include <iostream>
#include <algorithm>
#include <ctime>
#include <ratio>
#include <chrono>

namespace mini_cnn
{
	static inline nn_float vec_dot(const nn_float *nn_restrict x_vec, const nn_float *nn_restrict y_vec, nn_int len)
	{
		nn_float res = 0;
		for (nn_int i = 0; i < len; ++i)
		{
			res += x_vec[i] * y_vec[i];
		}
		return res;
	}

	// m: = m1 * m2
	//
	// gemm (general matrix multiply matrix) 
	// m1: h1 X w1
	// m2: h2 X w2, and m2 is transposed
	// m : h1 X h2
	static inline void gemm(const nn_float *nn_restrict m1, nn_int w1, nn_int h1
		, const nn_float *nn_restrict m2, nn_int w2, nn_int h2
		, nn_float *nn_restrict m, nn_int w, nn_int h)
	{
		nn_assert(w1 == w2);
		nn_assert(h1 == h && h2 == w);
		for (nn_int i = 0; i < h; ++i)
		{
			const nn_float *nn_restrict pm1 = m1 + i * w1;
			nn_float *nn_restrict pm = m + i * w;
			for (nn_int j = 0; j < w; ++j)
			{
				pm[j] = vec_dot(pm1, &m2[j * w2], w1);
			}
		}
	}

	// y := m * x
	// 
	// get vector by matrix multiply vector
	// m: matrix with shape of h X w
	// x, y: vector
	static inline void fo_mv_v(const nn_float *nn_restrict m, nn_int w, nn_int h
		, const nn_float *nn_restrict x
		, nn_float *nn_restrict y)
	{
		for (nn_int i = 0; i < h; ++i)
		{
			const nn_float *nn_restrict vec = m + i * w;
			nn_float res = 0;
			for (nn_int j = 0; j < w; ++j)
			{
				res += vec[j] * x[j];
			}
			y[i] = res;
		}
	}

	// m := x * y
	// 
	// get matrix by vector multiply vector
	// m: matrix with shape of h X w
	// x, y: vector
	static inline void fo_vv_m(const nn_float *nn_restrict x, nn_int h
		, const nn_float *nn_restrict y, nn_int w
		, nn_float *nn_restrict m)
	{
		for (nn_int i = 0; i < h; ++i)
		{
			nn_float *nn_restrict vec = m + i * w;
			nn_float xi = x[i];
			for (nn_int j = 0; j < w; ++j)
			{
				vec[j] = xi * y[j];
			}
		}
	}

}
#endif //__FAST_MATRIX_OPERATION_H__
