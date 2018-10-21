#ifndef __COMMON_DEF_H__
#define __COMMON_DEF_H__

#include <vector>
#include <cassert>

namespace mini_cnn
{
	typedef int					int_t;
	typedef unsigned int		uint_t;

	typedef float				float_t;

	typedef std::vector<int_t>	index_vec;

	const float_t cEPSILON = 1e-4f;

#ifndef PI
	const float_t cPI = 3.141592653589793f;
#endif

	const float_t cZero = (float_t)0.0;
	const float_t cOne = (float_t)1.0;

	const float_t cMIN_FLOAT = std::numeric_limits<float_t>::min();
	const float_t cMAX_FLOAT = std::numeric_limits<float_t>::max();

#ifdef _DEBUG
	#define nn_assert(cond) (void)(assert_break(cond))
	void assert_break(bool cond)
	{
		if (!cond)
		{
			std::cout << _CRT_WIDE(__FILE__) << __LINE__ << std::endl;
		}
		assert(cond);
	}
#else
	#define nn_assert(cond) ((void)0)
#endif

// memory alignment
#ifndef ALIGN
#if defined(__GNUC__)    // GCC
#    define ALIGN(n)    __attribute__((aligned(n)))
#elif defined( _MSC_VER ) // VC
#    define ALIGN(n)    __declspec(align(n))
#  endif
#endif // ALIGN

#define ALIGN_SIZE 32

#define USE_AVX
}
#endif // __COMMON_DEF_H__
