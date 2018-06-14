#ifndef __UTILS_H__
#define __UTILS_H__

#include <random>

#include "types.h"
#include "math/vectorn.h"
#include "math/matrixmxn.h"
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

typedef VectorN (*ActiveFunc)(const VectorN& vec);

inline VectorN Sigmoid(const VectorN& vec)
{
	VectorN retV(vec);
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] = 1.0f / (1.0f + exp(-vec[i]));
	}
	return retV;
}

inline VectorN SigmoidPrime(const VectorN& vec)
{
	VectorN retV(vec);
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		auto f = 1.0f / (1.0f + exp(-vec[i]));
		retV[i] = f * (1.0f - f);
	}
	return retV;
}

inline VectorN Relu(const VectorN& vec)
{
	VectorN retV(vec);
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] = vec[i] > 0.0f ? vec[i] : 0.0f;
	}
	return retV;
}

inline VectorN ReluPrime(const VectorN& vec)
{
	VectorN retV(vec);
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] = vec[i] > 0.0f ? 1.0f : 0.0f;	
	}
	return retV;
}

inline VectorN Softmax(const VectorN& vec)
{
	VectorN retV(vec);
	uint32_t idx = retV.ArgMax();
	float maxv = retV[idx];
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] -= maxv;
	}

	float sum = 0;
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] = exp(vec[i]);
		sum += retV[i];
	}

	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] /= sum;
	}
	return retV;
}

inline VectorN SoftmaxPrime(const VectorN& vec)
{
	VectorN retV(vec);
	uint32_t idx = retV.ArgMax();
	float maxv = retV[idx];
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] -= maxv;
	}

	float sum = 0;
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		retV[i] = exp(vec[i]);
		sum += retV[i];
	}

	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		float v = retV[i] / sum;
		retV[i] = v - v * v;
	}
	return retV;
}
#endif //__UTILS_H__
