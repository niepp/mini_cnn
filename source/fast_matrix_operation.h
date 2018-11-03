#ifndef __FAST_MATRIX_OPERATION_H__
#define __FAST_MATRIX_OPERATION_H__

#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

#include <Eigen/Dense>
using namespace Eigen;

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

#define nn_BLAS

#ifdef nn_BLAS
	typedef Map<Matrix<double, Dynamic, Dynamic, RowMajor>, AlignmentType::Unaligned> eigenMat_d_row_a32;
	typedef Map<Matrix<float, Dynamic, Dynamic, RowMajor>, AlignmentType::Unaligned> eigenMat_f_row_a32;
	typedef Map<VectorXd, AlignmentType::Unaligned> eigenVec_d_a32;
	typedef Map<VectorXf, AlignmentType::Unaligned> eigenVec_f_a32;

	// z := m't * x
	//
	// matrix multiply vector
	// m't: matrix m's transpose
	// x, z: vector
	static inline void fo_mtv_v(double *m, int w, int h
		, double *x
		, double *z)
	{
		eigenMat_d_row_a32 _m(m, h, w);
		eigenVec_d_a32 _x(x, h);
		eigenVec_d_a32 _z(z, w);
		_z = _m.transpose() * _x;
		z = _z.data();
	}

	static inline void fo_mtv_v(float *m, int w, int h
		, float *x
		, float *z)
	{
		eigenMat_f_row_a32 _m(m, h, w);
		eigenVec_f_a32 _x(x, h);
		eigenVec_f_a32 _z(z, w);
		_z = _m.transpose() * _x;
		z = _z.data();
	}

	// z := m * x + y
	//
	// matrix multiply vector
	// m: matrix
	// x, y, z: vector
	static inline void fo_mvv_v(double *m, int w, int h
		, double *x
		, double *y
		, double *z)
	{
		eigenMat_d_row_a32 _m(m, h, w);
		eigenVec_d_a32 _x(x, w);
		eigenVec_d_a32 _y(y, h);
		eigenVec_d_a32 _z(z, h);
		_z = _m * _x + _y;
		z = _z.data();
	}

	static inline void fo_mvv_v(float *m, int w, int h
		, float *x
		, float *y
		, float *z)
	{
		eigenMat_f_row_a32 _m(m, h, w);
		eigenVec_f_a32 _x(x, w);
		eigenVec_f_a32 _y(y, h);
		eigenVec_f_a32 _z(z, h);
		_z = _m * _x + _y;
		z = _z.data();
	}

	// m := x * y
	// 
	// get matrix by vector multiply vector
	// m: matrix
	// x, y: vector
	static inline void fo_vv_m(double *x, int w
		, double *y, int h
		, double *m)
	{
		eigenVec_d_a32 _x(x, w);
		eigenVec_d_a32 _y(y, h);
		eigenMat_d_row_a32 _m(m, w, h);
		_m = _x * _y.transpose();
		m = _m.data();
	}

	static inline void fo_vv_m(float *x, int w
		, float *y, int h
		, float *m)
	{
		eigenVec_f_a32 _x(x, w);
		eigenVec_f_a32 _y(y, h);
		eigenMat_f_row_a32 _m(m, w, h);
		_m = _x * _y.transpose();
		m = _m.data();
	}

	// m: = m1 * m2
	//
	// gemm (general matrix multiply matrix) 
	// m1: h1 X w1
	// m2: h2 X w2
	static inline void ge_mm(float *m1, int w1, int h1
		, float *m2, int w2, int h2
		, float *m, int w, int h)
	{
		nn_assert(w1 == h2);
		nn_assert(h1 == h && w2 == w);
		eigenMat_f_row_a32 _m1(m1, h1, w1);
		eigenMat_f_row_a32 _m2(m2, h2, w2);
		eigenMat_f_row_a32 _m(m, h, w);
		_m = _m1 * _m2;
		m = _m.data();
	}

#else
	static inline void fo_mtv_v(nn_float *nn_restrict m, nn_int w, nn_int h
	, nn_float *nn_restrict x
	, nn_float *nn_restrict z)
	{
		for (nn_int i = 0; i < w; ++i)
		{
			nn_float dot = 0;
			for (nn_int j = 0; j < h; ++j)
			{
				dot += m[i + j * w] * x[j];
			}
			z[i] = dot;
		}
	}

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

	static inline void fo_vv_m(nn_float *nn_restrict x, nn_int w, nn_float *nn_restrict y, nn_int h, nn_float *nn_restrict m)
	{
		for (nn_int i = 0; i < h; ++i)
		{
			nn_float *nn_restrict vec_m = &m[i * w];
			nn_float yi = y[i];
			for (nn_int j = 0; j < w; ++j)
			{
				vec_m[j] = x[j] * yi;
			}
		}
	}
#endif




	//inline unsigned char* align_address(size_t address, int align_size)
	//{
	//	return (unsigned char*)((address + align_size - 1) & (-align_size));
	//}

	//inline void* align_malloc(size_t size, size_t align_size)
	//{
	//	unsigned char* mptr = (unsigned char*)::malloc(size + sizeof(void*)+align_size);
	//	if (!mptr)
	//	{
	//		return nullptr;
	//	}
	//	unsigned char* aptr = align_address((size_t)mptr + sizeof(void*), align_size);
	//	unsigned char**p = (unsigned char**)((size_t)aptr - sizeof(void*));
	//	*p = mptr;
	//	return aptr;
	//}

	//inline long long get_now_ms()
	//{
	//	auto tp_now = std::chrono::high_resolution_clock::now();
	//	return std::chrono::duration_cast<std::chrono::milliseconds>(tp_now.time_since_epoch()).count();
	//}

	//static inline void gemv_native(double _alpha, double *_m, int _w, int _h
	//	, double *_x
	//	, double _beta, double *_y
	//	, double *_z)
	//{
	//	for (int i = 0; i < _h; ++i)
	//	{
	//		double dot = 0;
	//		for (int j = 0; j < _w; ++j)
	//		{
	//			dot += _alpha * _m[j + i * _w] * _x[j];
	//		}
	//		_z[i] = dot;
	//	}
	//}

	//static void eigen_test()
	//{
	//	const int TestCase = 1000;
	//	const int M = 1000, N = 1200;
	//	double _alpha = 1.0;
	//	double *_m = (double*)align_malloc(sizeof(double)* M * N, 32);

	//	for (int i = 0; i < N; ++i) {
	//		for (int j = 0; j < M; ++j) {
	//			_m[j + i * M] = 1.0 * (i + j) / (M * N);
	//		}
	//	}

	//	int _w = M;
	//	int _h = N;
	//	double *_x = (double*)align_malloc(sizeof(double)* M, 32);
	//	for (int i = 0; i < M; ++i) {
	//		_x[i] = 1.0 * i / M;
	//	}

	//	double _beta = 0;
	//	double *_y = (double*)align_malloc(sizeof(double)* M, 32);
	//	double *_z = (double*)align_malloc(sizeof(double)* N, 32);
	//	for (int i = 0; i < M; ++i) {
	//		_y[i] = 1.0 - (1.0 * i / M);
	//	}
	//	for (int i = 0; i < N; ++i) {
	//		_z[i] = 0;
	//	}

	//	double t0 = get_now_ms();

	//	for (int i = 0; i < TestCase; ++i) {
	//		gemv(1.0, _m, _w, _h, _x, 0, _y, _z);
	//	}

	//	double t1 = get_now_ms();

	//	for (int i = 0; i < TestCase; ++i) {
	//		gemv_native(1.0, _m, _w, _h, _x, 0, _y, _z);
	//	}

	//	double t2 = get_now_ms();

	//	std::cout << "gemv:   " << t1 - t0 << std::endl;
	//	std::cout << "native: " << t2 - t1 << std::endl;

	//	double sum = 0;
	//	for (int i = 0; i < M; ++i) {
	//		sum += _z[i];
	//	}

	//	std::cout << "sum: " << sum << std::endl;

	//}


}
#endif //__FAST_MATRIX_OPERATION_H__
