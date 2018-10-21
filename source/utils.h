#ifndef __UTILS_H__
#define __UTILS_H__

#include <cassert>
#include <random>
#include <chrono>

namespace mini_cnn
{
class normal_random
{
private:
	float_t m_mean;
	float_t m_stdev;
	int_t m_truncated;
public:
	std::mt19937_64 m_generator;
	std::normal_distribution<float_t> m_distribution;
public:
	normal_random(std::mt19937_64 generator, float_t mean = 0, float_t stdev = 1.0, int_t truncated = 0) :
		m_mean(mean), m_stdev(stdev), m_truncated(truncated), m_generator(generator), m_distribution(mean, stdev)
	{
	}

	float_t get_random()
	{
		if (m_truncated <= 0)
		{
			return m_distribution(m_generator);
		}
		else
		{
			float_t r = m_mean;
			do
			{
				r = m_distribution(m_generator);
			}
			while (abs(r - m_mean) >= m_truncated * m_stdev);
			return r;
		}
	}
};

class uniform_random
{
public:
	std::mt19937_64 m_generator;
	std::uniform_real_distribution<float_t> m_distribution;
public:
	uniform_random(std::mt19937_64 generator, float_t min, float_t max) : m_generator(generator), m_distribution(min, max)
	{
	}

	float_t get_random()
	{
		return m_distribution(m_generator);
	}

};

inline uint_t get_now()
{
	auto tp_now = std::chrono::steady_clock::now();
	auto ms_now = std::chrono::time_point_cast<std::chrono::milliseconds>(tp_now);
	auto t_now = ms_now.time_since_epoch();
	return static_cast<uint_t>(t_now.count());
}

inline bool f_is_valid(float_t f)
{
	return f == f;
}

inline unsigned char* align_address(size_t address, int align_size)
{
	return (unsigned char*)((address + align_size - 1) & (-align_size));
}

inline void* align_malloc(int_t size)
{
	unsigned char* mptr = (unsigned char*)::malloc(size + sizeof(void*) + ALIGN_SIZE);
	if (!mptr)
	{
		return nullptr;
	}
	unsigned char* aptr = align_address((size_t)mptr + sizeof(void*), ALIGN_SIZE);
	unsigned char**p = (unsigned char**)((size_t)aptr - sizeof(void*));
	*p = mptr;
	return aptr;
}

inline void align_free(void *aptr)
{
	if (aptr != nullptr)
	{
		unsigned char**p = (unsigned char**)((size_t)aptr - sizeof(void*));
		::free(*p);
	}
}

typedef void (*active_func)(const varray &v, varray &retv);

inline void sigmoid(const varray &v, varray &retv)
{
	int_t len = v.size();
	nn_assert(len == retv.size());
	for (int_t i = 0; i < len; ++i)
	{
		retv[i] = cOne / (cOne + exp(-v[i]));
	}
}

inline void deriv_sigmoid(const varray &v, varray &retv)
{
	int_t len = v.size();
	nn_assert(len == retv.size());
	for (int_t i = 0; i < len; ++i)
	{
		float_t t = cOne / (cOne + exp(-v[i]));
		retv[i] = t * (cOne - t);
	}
}

inline void relu(const varray &v, varray &retv)
{
	int_t len = v.size();
	nn_assert(len == retv.size());
	for (int_t i = 0; i < len; ++i)
	{
		retv[i] = v[i] > 0 ? v[i] : 0;
	}
}

inline void deriv_relu(const varray &v, varray &retv)
{
	int_t len = v.size();
	nn_assert(len == retv.size());
	for (int_t i = 0; i < len; ++i)
	{
		retv[i] = v[i] > 0 ? cOne : 0;
	}
}

inline void softmax(const varray &v, varray &retv)
{
	int_t len = v.size();
	nn_assert(len == retv.size());
	int_t idx = v.arg_max();
	float_t maxv = v[idx];
	for (int_t i = 0; i < len; ++i)
	{
		retv[i] = v[i] - maxv;
	}
	float_t s = 0;
	for (int_t i = 0; i < len; ++i)
	{
		retv[i] = exp(retv[i]);
		s += retv[i];
	}
	for (int_t i = 0; i < len; ++i)
	{
		retv[i] /= s;
	}
}

}

#endif //__UTILS_H__
