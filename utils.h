#ifndef __UTILS_H__
#define __UTILS_H__

#include <random>
#include <chrono>

#include "types.h"
#include "math/vectorn.h"
#include "math/matrixmxn.h"
#include "math/matrix3d.h"
#include "math/mathdef.h"
namespace mini_cnn
{
class NormalRandom
{
private:
	Float m_mean;
	Float m_stdev;
	Int m_nTruncated;
public:
	std::mt19937_64 m_generator;
	std::normal_distribution<Float> m_distribution;
public:
	NormalRandom(std::mt19937_64 generator, Float mean = 0, Float stdev = 1.0, Int nTruncated = 0) :
		m_mean(mean), m_stdev(stdev), m_nTruncated(nTruncated), m_generator(generator), m_distribution(mean, stdev)
	{
	}

	Float GetRandom()
	{
		if (m_nTruncated <= 0)
		{
			return m_distribution(m_generator);
		}
		else
		{
			Float r = m_mean;
			do
			{
				r = m_distribution(m_generator);
			}
			while (abs(r - m_mean) >= m_nTruncated * m_stdev);
			return r;
		}
	}
};

class UniformRandom
{
public:
	std::mt19937_64 m_generator;
	std::uniform_real_distribution<Float> m_distribution;
public:
	UniformRandom(std::mt19937_64 generator, Float min, Float max) : m_generator(generator), m_distribution(min, max)
	{
	}

	Float GetRandom()
	{
		return m_distribution(m_generator);
	}

};

inline uInt GetNow()
{
	auto tp_now = std::chrono::steady_clock::now();
	auto ms_now = std::chrono::time_point_cast<std::chrono::milliseconds>(tp_now);
	auto t_now = ms_now.time_since_epoch();
	return static_cast<uInt>(t_now.count());
}

inline bool f_is_valid(Float f)
{
	return f == f;
}

typedef void(*ActiveFunc)(const VectorN& vec, VectorN& retV);

inline void Sigmoid(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = (Float)(1.0) / ((Float)(1.0) + exp(-vec[i]));
	}
}

inline void SigmoidPrime(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		Float f = (Float)(1.0) / ((Float)(1.0) + exp(-vec[i]));
		retV[i] = f * ((Float)(1.0) - f);
	}
}

inline void Tanh(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		const Float ep = std::exp(vec[i]);
		const Float em = std::exp(-vec[i]);
		retV[i] = (ep - em) / (ep + em);
	}
}

inline void TanhPrime(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		const Float ep = std::exp(vec[i]);
		const Float em = std::exp(-vec[i]);
		Float f = (ep - em) / (ep + em);
		retV[i] = (Float)(1.0) - f * f;
	}
}

inline void Relu(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] > 0.0 ? vec[i] : 0.0;
	}
}

inline void ReluPrime(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] > 0.0 ? 1.0 : 0.0;
	}
}

inline void Softmax(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	uInt idx = vec.ArgMax();
	Float maxv = vec[idx];
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] - maxv;
	}

	Float sum = 0;
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = exp(retV[i]);
		sum += retV[i];
	}

	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] /= sum;
	}
}

inline void SoftmaxPrime(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	uInt idx = vec.ArgMax();
	Float maxv = vec[idx];
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] - maxv;
	}

	Float sum = 0;
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = exp(retV[i]);
		sum += retV[i];
	}

	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		Float v = retV[i] / sum;
		retV[i] = v - v * v;
	}
}

//
typedef void(*MatActiveFunc)(const Matrix3D& mat, Matrix3D &retMat);

inline void Sigmoid(const Matrix3D &mat, Matrix3D &retMat)
{
	assert(mat.Width() == retMat.Width()
		&& mat.Height() == retMat.Height()
		&& mat.Depth() == retMat.Depth());

	for (uInt k = 0; k < mat.Depth(); ++k)
	{
		for (uInt i = 0; i < mat.Width(); ++i)
		{
			for (uInt j = 0; j < mat.Height(); ++j)
			{
				Float v = mat(i, j, k);
				retMat(i, j, k) = (Float)(1.0) / ((Float)(1.0) + exp(-v));
			}
		}
	}
}

inline void SigmoidPrime(const Matrix3D &mat, Matrix3D &retMat)
{
	assert(mat.Width() == retMat.Width()
		&& mat.Height() == retMat.Height()
		&& mat.Depth() == retMat.Depth());

	for (uInt k = 0; k < mat.Depth(); ++k)
	{
		for (uInt i = 0; i < mat.Width(); ++i)
		{
			for (uInt j = 0; j < mat.Height(); ++j)
			{
				Float v = mat(i, j, k);
				Float f = (Float)(1.0) / ((Float)(1.0) + exp(-v));
				retMat(i, j, k) = f * ((Float)(1.0) - f);
			}
		}
	}
}

inline void Tanh(const Matrix3D &mat, Matrix3D &retMat)
{
	assert(mat.Width() == retMat.Width()
		&& mat.Height() == retMat.Height()
		&& mat.Depth() == retMat.Depth());

	for (uInt k = 0; k < mat.Depth(); ++k)
	{
		for (uInt i = 0; i < mat.Width(); ++i)
		{
			for (uInt j = 0; j < mat.Height(); ++j)
			{
				Float v = mat(i, j, k);
				const Float ep = std::exp(v);
				const Float em = std::exp(-v);
				retMat(i, j, k) = (ep - em) / (ep + em);
			}
		}
	}
}

inline void TanhPrime(const Matrix3D &mat, Matrix3D &retMat)
{
	assert(mat.Width() == retMat.Width()
		&& mat.Height() == retMat.Height()
		&& mat.Depth() == retMat.Depth());

	for (uInt k = 0; k < mat.Depth(); ++k)
	{
		for (uInt i = 0; i < mat.Width(); ++i)
		{
			for (uInt j = 0; j < mat.Height(); ++j)
			{
				Float v = mat(i, j, k);
				const Float ep = std::exp(v);
				const Float em = std::exp(-v);
				Float f = (ep - em) / (ep + em);
				retMat(i, j, k) = (Float)(1.0) - f * f;
			}
		}
	}
}

inline void Relu(const Matrix3D &mat, Matrix3D &retMat)
{
	assert(mat.Width() == retMat.Width()
		&& mat.Height() == retMat.Height()
		&& mat.Depth() == retMat.Depth());

	for (uInt k = 0; k < mat.Depth(); ++k)
	{
		for (uInt i = 0; i < mat.Width(); ++i)
		{
			for (uInt j = 0; j < mat.Height(); ++j)
			{
				Float v = mat(i, j, k);
				retMat(i, j, k) = v > 0.0 ? v : 0.0;
			}
		}
	}
}

inline void ReluPrime(const Matrix3D &mat, Matrix3D &retMat)
{
	assert(mat.Width() == retMat.Width()
		&& mat.Height() == retMat.Height()
		&& mat.Depth() == retMat.Depth());

	for (uInt k = 0; k < mat.Depth(); ++k)
	{
		for (uInt i = 0; i < mat.Width(); ++i)
		{
			for (uInt j = 0; j < mat.Height(); ++j)
			{
				Float v = mat(i, j, k);
				retMat(i, j, k) = v > 0.0 ? 1.0 : 0.0;
			}
		}
	}
}

}

#endif //__UTILS_H__
