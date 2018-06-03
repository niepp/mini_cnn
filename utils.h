#pragma once
#include <random>

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
