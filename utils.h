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

typedef VectorN& (*ActiveFunc)(VectorN& vec);

inline VectorN& Sigmoid(VectorN& vec)
{
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		vec[i] = 1.0f / (1.0f + exp(-vec[i]));
	}
	return vec;
}

inline VectorN& SigmoidPrime(VectorN& vec)
{
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		auto f = 1.0f / (1.0f + exp(-vec[i]));
		vec[i] = f * (1.0f - f);
	}
	return vec;
}

inline VectorN& Relu(VectorN& vec)
{
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		if (vec[i] < 0.0f)
		{
			vec[i] = 0.0f;
		}
	}
	return vec;
}

inline VectorN& ReluPrime(VectorN& vec)
{
	for (unsigned int i = 0; i < vec.GetSize(); ++i)
	{
		if (vec[i] > 0.0f)
		{ 
			vec[i] = 1.0f;
		}
		else
		{
			vec[i] = 0.0f;
		}
	}
	return vec;
}

#endif //__UTILS_H__
