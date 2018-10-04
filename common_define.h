#ifndef __COMMON_DEF_H__
#define __COMMON_DEF_H__

namespace mini_cnn
{
	typedef int					int_t;
	typedef unsigned int		uint_t;

	typedef double				float_t;

	typedef std::vector<int_t>	index_vec;

	const float_t cEPSILON = 1e-4;

#ifndef PI
	const float_t cPI = 3.141592653589793;
#endif

	const float_t cOne = (float_t)1.0;

	const float_t cMIN_FLOAT = std::numeric_limits<float_t>::min();
	const float_t cMAX_FLOAT = std::numeric_limits<float_t>::max();

#ifdef NDEBUG
	#define nn_assert(cond) ((void)0)
#else
	#define nn_assert(cond) (void)(assert_break(cond))
	void assert_break(bool cond)
	{
		if (!cond)
		{
			std::cout << _CRT_WIDE(__FILE__) << __LINE__ << std::endl;
		}
		assert(cond);
	}

#endif

}
#endif // __COMMON_DEF_H__
