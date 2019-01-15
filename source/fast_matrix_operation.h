#ifndef __FAST_MATRIX_OPERATION_H__
#define __FAST_MATRIX_OPERATION_H__

#include <iostream>
#include <algorithm>
#include <ctime>
#include <ratio>
#include <chrono>

//#define USE_BLAS

#ifdef USE_BLAS
#include "mkl.h"
#endif

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

#ifdef USE_BLAS
	/*
		blas_gemm
	*/
	template<typename T>
	static inline void blas_gemm(const T *nn_restrict mat_a, nn_int h1, nn_int w1
		, const T *nn_restrict mat_b, nn_int h2, nn_int w2
		, T *nn_restrict mat_c, nn_int h, nn_int w);

	template<>
	static inline void blas_gemm<float>(const float *nn_restrict mat_a, nn_int h1, nn_int w1
		, const float *nn_restrict mat_b, nn_int h2, nn_int w2
		, float *nn_restrict mat_c, nn_int h, nn_int w)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			h1, w, w1, 1.0, mat_a, w1, mat_b, w2, 0, mat_c, w);
	}
	template<>
	static inline void blas_gemm<double>(const double *nn_restrict mat_a, nn_int h1, nn_int w1
		, const double *nn_restrict mat_b, nn_int h2, nn_int w2
		, double *nn_restrict mat_c, nn_int h, nn_int w)
	{
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			h1, w, w1, 1.0, mat_a, w1, mat_b, w2, 0, mat_c, w);
	}

	/*
		blas_gemv
	*/
	template<typename T>
	static inline void blas_gemv(const T *nn_restrict m, nn_int w, nn_int h
		, const T *nn_restrict x
		, T *nn_restrict y);

	template<>
	static inline void blas_gemv<float>(const float *nn_restrict m, nn_int w, nn_int h
		, const float *nn_restrict x
		, float *nn_restrict y)
	{
		cblas_sgemv(CblasRowMajor, CblasNoTrans,
			h, w, 1.0, m, w, x, 1, 0.0, y, 1);
	}
	template<>
	static inline void blas_gemv<double>(const double *nn_restrict m, nn_int w, nn_int h
		, const double *nn_restrict x
		, double *nn_restrict y)
	{
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			h, w, 1.0, m, w, x, 1, 0.0, y, 1);
	}

	/*
		blas_ger
	*/
	template<typename T>
	static inline void blas_ger(const T *nn_restrict x, nn_int h
		, const T *nn_restrict y, nn_int w
		, T *nn_restrict m);

	template<>
	static inline void blas_ger(const float *nn_restrict x, nn_int h
		, const float *nn_restrict y, nn_int w
		, float *nn_restrict m)
	{
		cblas_sger(CblasRowMajor, h, w, 1.0, x, 1, y, 1, m, w);
	}
	template<>
	static inline void blas_ger(const double *nn_restrict x, nn_int h
		, const double *nn_restrict y, nn_int w
		, double *nn_restrict m)
	{
		cblas_dger(CblasRowMajor, h, w, 1.0, x, 1, y, 1, m, w);
	}

	/*
		vector add / scale
	*/
	template<typename T>
	static inline void blas_gvv(const T *nn_restrict x, nn_int nx
		, const T alpha
		, T *nn_restrict y
		, nn_int ny);

	template<>
	static inline void blas_gvv(const float *nn_restrict x, nn_int nx
		, const float alpha
		, float *nn_restrict y
		, nn_int ny)
	{
		cblas_saxpy(nx, alpha, x, 1, y, 1);
	}
	template<>
	static inline void blas_gvv(const double *nn_restrict x, nn_int nx
		, const double alpha
		, double *nn_restrict y
		, nn_int ny)
	{
		cblas_daxpy(nx, alpha, x, 1, y, 1);
	}

#endif

	// mat_c : = mat_a * mat_b
	//
	// gemm (general matrix multiply matrix) 
	// mat_a : h1 X w1
	// mat_b : h2 X w2, and m2 is transposed
	// mat_c : h1 X h2
	static inline void gemm(const nn_float *nn_restrict mat_a, nn_int h1, nn_int w1
		, const nn_float *nn_restrict mat_b, nn_int h2, nn_int w2
		, nn_float *nn_restrict mat_c, nn_int h, nn_int w)
	{
		nn_assert(w1 == w2);
		nn_assert(h1 == h && h2 == w);

#ifdef USE_BLAS
		blas_gemm<nn_float>(mat_a, h1, w1
			, mat_b, h2, w2
			, mat_c, h, w);
#else
		for (nn_int i = 0; i < h; ++i)
		{
			const nn_float *nn_restrict pa = mat_a + i * w1;
			nn_float *nn_restrict pc = mat_c + i * w;
			for (nn_int j = 0; j < w; ++j)
			{
				pc[j] = vec_dot(pa, &mat_b[j * w2], w1);
			}
		}
#endif

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
#ifdef USE_BLAS
		blas_gemv<nn_float>(m, w, h, x, y);
#else
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
#endif
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
#ifdef USE_BLAS
		blas_ger<nn_float>(x, h, y, w, m);
#else
		for (nn_int i = 0; i < h; ++i)
		{
			nn_float *nn_restrict vec = m + i * w;
			nn_float xi = x[i];
			for (nn_int j = 0; j < w; ++j)
			{
				vec[j] = xi * y[j];
			}
		}
#endif
	}

	// y := alpha * x + y
	// 
	// get matrix by vector multiply vector
	// alpha: matrix with shape of h X w
	// x, y: vector
	static inline void fo_vv(const nn_float *nn_restrict x, nn_int nx
		, const nn_float alpha
		, nn_float *nn_restrict y
		, nn_int ny)
	{
		nn_assert(nx == ny);
#ifdef USE_BLAS
		blas_gvv<nn_float>(x, nx, alpha, y, ny);
#else
		for (nn_int i = 0; i < nx; ++i)
		{
			y[i] += x[i] * alpha;
		}
#endif
	}

}
#endif //__FAST_MATRIX_OPERATION_H__
