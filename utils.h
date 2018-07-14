#ifndef __UTILS_H__
#define __UTILS_H__

#include <random>

#include "types.h"
#include "math/vectorn.h"
#include "math/matrixmxn.h"
#include "math/matrix3d.h"
#include "math/mathdef.h"

class NormalRandom
{
public:
	std::mt19937_64 m_generator;
	std::normal_distribution<float> m_distribution;
public:
	NormalRandom(float mean, float stdev) : m_distribution(mean, stdev)
	{
	}

	float GetRandom()
	{
		return m_distribution(m_generator);
	}

};

typedef void(*ActiveFunc)(const VectorN& vec, VectorN& retV);

inline void Sigmoid(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = 1.0f / (1.0f + exp(-vec[i]));
	}
}

inline void SigmoidPrime(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		auto f = 1.0f / (1.0f + exp(-vec[i]));
		retV[i] = f * (1.0f - f);
	}
}

inline void Relu(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] > 0.0f ? vec[i] : 0.0f;
	}
}

inline void ReluPrime(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] > 0.0f ? 1.0f : 0.0f;
	}
}

inline void Softmax(const VectorN& vec, VectorN& retV)
{
	assert(vec.GetSize() == retV.GetSize());
	uint32_t idx = vec.ArgMax();
	float maxv = vec[idx];
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] - maxv;
	}

	float sum = 0;
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
	uint32_t idx = vec.ArgMax();
	float maxv = vec[idx];
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = vec[i] - maxv;
	}

	float sum = 0;
	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		retV[i] = exp(retV[i]);
		sum += retV[i];
	}

	for (unsigned int i = 0; i < retV.GetSize(); ++i)
	{
		float v = retV[i] / sum;
		retV[i] = v - v * v;
	}
}
#endif //__UTILS_H__
