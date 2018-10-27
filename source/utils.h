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
	nn_float m_mean;
	nn_float m_stdev;
	nn_int m_truncated;
public:
	std::mt19937_64 m_generator;
	std::normal_distribution<nn_float> m_distribution;
public:
	normal_random(std::mt19937_64 generator, nn_float mean = 0, nn_float stdev = 1.0, nn_int truncated = 0) :
		m_mean(mean), m_stdev(stdev), m_truncated(truncated), m_generator(generator), m_distribution(mean, stdev)
	{
	}

	nn_float get_random()
	{
		if (m_truncated <= 0)
		{
			return m_distribution(m_generator);
		}
		else
		{
			nn_float r = m_mean;
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
	std::uniform_real_distribution<nn_float> m_distribution;
public:
	uniform_random(std::mt19937_64 generator, nn_float min, nn_float max) : m_generator(generator), m_distribution(min, max)
	{
	}

	nn_float get_random()
	{
		return m_distribution(m_generator);
	}

};

inline nn_float get_now_ms()
{
	auto tp_now = std::chrono::high_resolution_clock::now();
	auto ms_now = std::chrono::time_point_cast<std::chrono::milliseconds>(tp_now);
	auto t_now = ms_now.time_since_epoch();
	return static_cast<nn_float>(t_now.count());
}

inline bool f_is_valid(nn_float f)
{
	return f == f;
}

inline unsigned char* align_address(size_t address, int align_size)
{
	return (unsigned char*)((address + align_size - 1) & (-align_size));
}

inline void* align_malloc(nn_int size)
{
	unsigned char* mptr = (unsigned char*)::malloc(size + sizeof(void*) + nn_align_size);
	if (!mptr)
	{
		return nullptr;
	}
	unsigned char* aptr = align_address((size_t)mptr + sizeof(void*), nn_align_size);
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
	nn_int len = v.size();
	nn_assert(len == retv.size());
	for (nn_int i = 0; i < len; ++i)
	{
		retv[i] = cOne / (cOne + exp(-v[i]));
	}
}

inline void deriv_sigmoid(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());

	const nn_float * __restrict src = &v[0];
	nn_float * __restrict dst = &retv[0];
	for (nn_int i = 0; i < len; ++i)
	{
		nn_float t = cOne / (cOne + exp(-src[i]));
		dst[i] = t * (cOne - t);
	}
}

inline void relu(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());
	for (nn_int i = 0; i < len; ++i)
	{
		retv[i] = v[i] > 0 ? v[i] : 0;
	}
}

inline void deriv_relu(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());
	const nn_float * __restrict src = &v[0];
	nn_float * __restrict dst = &retv[0];
	for (nn_int i = 0; i < len; ++i)
	{
		dst[i] = src[i] > 0 ? cOne : 0;
	}
}

inline void softmax(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());
	nn_int idx = v.arg_max();
	nn_float maxv = v[idx];
	for (nn_int i = 0; i < len; ++i)
	{
		retv[i] = v[i] - maxv;
	}
	nn_float s = 0;
	for (nn_int i = 0; i < len; ++i)
	{
		retv[i] = exp(retv[i]);
		s += retv[i];
	}
	s = (nn_float)(1.0) / s;
	for (nn_int i = 0; i < len; ++i)
	{
		retv[i] *= s;
	}
}

}

#endif //__UTILS_H__
