#ifndef __GLOBAL_SETTING_H__
#define __GLOBAL_SETTING_H__

#include <random>

namespace mini_cnn
{

class global_setting
{
public:
	static std::mt19937_64 m_rand_generator;
};

}

#endif //__GLOBAL_SETTING_H__
