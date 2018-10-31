#ifndef __COMMON_DEF_H__
#define __COMMON_DEF_H__

#include <vector>
#include <cassert>

namespace mini_cnn
{

	typedef int					nn_int;
	typedef unsigned int		nn_uint;

// gradient checker need high percise
#ifdef GRADIENT_CHECKER
	typedef double				nn_float;
#else
	typedef float				nn_float;
#endif

	typedef std::vector<nn_int>	index_vec;

	const nn_float cEpsilon = 1e-4f;

#ifndef PI
	const nn_float cPI = 3.141592653589793f;
#endif

	const nn_float cZero = (nn_float)0.0;
	const nn_float cOne = (nn_float)1.0;

	const nn_float cMinFloat = std::numeric_limits<nn_float>::min();
	const nn_float cMaxFloat = std::numeric_limits<nn_float>::max();

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
#ifndef nn_align
#if defined(__GNUC__)    // GCC
#    define nn_align(n)    __attribute__((aligned(n)))
#elif defined( _MSC_VER ) // VC
#    define nn_align(n)    __declspec(align(n))
#  endif
#endif // nn_align

#define nn_align_size 32

#define nn_restrict __restrict

}
#endif // __COMMON_DEF_H__
