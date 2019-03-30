#ifndef __UTILS_H__
#define __UTILS_H__

#include <cassert>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>
#include <future>

namespace mini_cnn
{

class normal_random
{
private:
	nn_float m_mean;
	nn_float m_stdev;
	nn_int m_truncated;
public:
	std::normal_distribution<nn_float> m_distribution;
public:
	normal_random(nn_float mean = 0, nn_float stdev = 1.0, nn_int truncated = 0) :
		m_mean(mean), m_stdev(stdev), m_truncated(truncated), m_distribution(mean, stdev)
	{
	}

	nn_float get_random()
	{
		if (m_truncated <= 0)
		{
			return m_distribution(global_setting::m_rand_generator);
		}
		else
		{
			nn_float r = m_mean;
			do
			{
				r = m_distribution(global_setting::m_rand_generator);
			}
			while (abs(r - m_mean) >= m_truncated * m_stdev);
			return r;
		}
	}
};

class uniform_random
{
public:
	std::uniform_real_distribution<nn_float> m_distribution;
public:
	uniform_random(nn_float min, nn_float max) : m_distribution(min, max)
	{
	}

	nn_float get_random()
	{
		return m_distribution(global_setting::m_rand_generator);
	}

};

inline long long get_now_ms()
{
	auto tp_now = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::milliseconds>(tp_now.time_since_epoch()).count();
}

inline unsigned char* align_address(size_t address, int align_size)
{
	return (unsigned char*)((address + align_size - 1) & (-align_size));
}

inline void* align_malloc(size_t size, int align_size)
{
	unsigned char* mptr = (unsigned char*)::malloc(size + sizeof(void*) + align_size);
	if (!mptr)
	{
		return nullptr;
	}
	unsigned char* aptr = align_address((size_t)mptr + sizeof(void*), align_size);
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

inline bool f_is_valid(nn_float f)
{
	return f == f;
}

inline bool is_valid(const varray &vec)
{
	nn_int len = vec.size();
	for (nn_int i = 0; i < len; ++i)
	{
		if (!f_is_valid(vec[i])) {
			return false;
		}
	}
	return true;
}

inline float fast_inv_sqrt(float x)
{
	float y = x;
	float x2 = x * 0.5f;
	int i = *(int*)&y;  // evil floating point bit level hacking
	i = 0x5f3759df - (i >> 1); // what the fuck?
	y = *(float*)&i;
	y = y * (1.5f - (x2 * y * y)); // 1st iteration
	// y = y * (1.5f - (x2 * y * y)); // 2nd iteration, this can be removed
	return y;
}

inline double fast_inv_sqrt(double x)
{
	double y = x;
	double x2 = y * 0.5;
	int64_t i = *(int64_t *) &y;
	// The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
	i = 0x5fe6eb50c7b537a9 - (i >> 1);
	y = *(double *)&i;
	y = y * (1.5 - (x2 * y * y));   // 1st iteration
	// y  = y * ( 1.5 - ( x2 * y * y ) ); // 2nd iteration, this can be removed
	return y;
}

inline float fast_exp(float x)
{
	int a = 185 * (int)x + 16249;
	a <<= 16;
	float f = *(reinterpret_cast<float*>(&a));
	return f;
}

inline double fast_exp(double x)
{
	double d;
	*(reinterpret_cast<int*>(&d) + 0) = 0;
	*(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * x + 1072632447);
	return d;
}

inline void transpose(const varray &mat, varray &retm)
{
	nn_assert(mat.dim() == 2);
	nn_assert(retm.dim() == 2);

	nn_assert(mat.width() == retm.height());
	nn_assert(mat.height() == retm.width());

	nn_int h = retm.height();
	nn_int w = retm.width();

	for (nn_int i = 0; i < h; ++i)
	{
		for (nn_int j = 0; j < w; ++j)
		{
			retm[i * w + j] = mat[j * h + i];
		}
	}
}

inline nn_int arg_max(const nn_float *nn_restrict vec, nn_int len)
{
	nn_int max_idx = 0;
	nn_float m = vec[0];
	for (nn_int i = 1; i < len; ++i)
	{
		if (vec[i] > m)
		{
			m = vec[i];
			max_idx = i;
		}
	}
	return max_idx;
}

void parallel_task(nn_int batch_size, nn_int task_count, std::function<void(nn_int, nn_int, nn_int)> func)
{
	nn_int nstep = (batch_size + task_count - 1) / task_count;
	std::vector<std::future<void>> futures;
	for (nn_int k = 0; k < task_count && k * nstep < batch_size; ++k)
	{
		nn_int begin = k * nstep;
		nn_int end = std::min(batch_size, begin + nstep);
		futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
			func(begin, end, k);
		})));
	}
	for (auto &future : futures)
	{
		future.get();
	}
}

typedef void (*active_func)(const varray &v, varray &retv);

inline void identity(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());

	const nn_float * nn_restrict src = &v[0];
	nn_float * nn_restrict dst = &retv[0];
	for (nn_int i = 0; i < len; ++i)
	{
		dst[i] = src[i];
	}
}

inline void deriv_identity(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());

	const nn_float * nn_restrict src = &v[0];
	nn_float * nn_restrict dst = &retv[0];
	for (nn_int i = 0; i < len; ++i)
	{
		dst[i] = cOne;
	}
}

inline void sigmoid(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());

	const nn_float * nn_restrict src = &v[0];
	nn_float * nn_restrict dst = &retv[0];
	for (nn_int i = 0; i < len; ++i)
	{
		dst[i] = cOne / (cOne + exp(-src[i]));
	}
}

inline void deriv_sigmoid(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());

	const nn_float * nn_restrict src = &v[0];
	nn_float * nn_restrict dst = &retv[0];
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

	const nn_float * nn_restrict src = &v[0];
	nn_float * nn_restrict dst = &retv[0];
	for (nn_int i = 0; i < len; ++i)
	{
		dst[i] = src[i] > 0 ? src[i] : 0;
	}
}

inline void deriv_relu(const varray &v, varray &retv)
{
	nn_int len = v.size();
	nn_assert(len == retv.size());

	const nn_float * nn_restrict src = &v[0];
	nn_float * nn_restrict dst = &retv[0];
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

	const nn_float * nn_restrict src = &v[0];
	nn_float * nn_restrict dst = &retv[0];
	for (nn_int i = 0; i < len; ++i)
	{
		dst[i] = exp(src[i] - maxv);
	}
	nn_float s = 0;
	for (nn_int i = 0; i < len; ++i)
	{
		s += dst[i];
	}
	s = (nn_float)(1.0) / s;
	for (nn_int i = 0; i < len; ++i)
	{
		dst[i] *= s;
	}
}

}

#endif //__UTILS_H__
