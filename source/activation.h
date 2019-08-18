#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <cassert>
#include <algorithm>

namespace mini_cnn
{

enum activation_type
{
	eNone,
	eIdentity,
	eSigmod,
	eTanh,
	eRelu,
	eSoftmax,
};

class activation_base
{
protected:
	activation_type m_act_type;
public:
	activation_base(activation_type act_type) 
		: m_act_type(act_type)
	{ 
	}

	activation_type act_type() const
	{
		return m_act_type;
	}

	virtual void f(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len) = 0;

	virtual void df(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len) = 0;

};

class activation_identity : public activation_base
{
public:
	activation_identity() : activation_base(activation_type::eIdentity)
	{
	}

	virtual void f(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		for (nn_int i = 0; i < len; ++i)
		{
			dst[i] = src[i];
		}
	}

	virtual void df(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		for (nn_int i = 0; i < len; ++i)
		{
			dst[i] = cOne;
		}
	}
};

class activation_sigmoid : public activation_base
{
public:
	activation_sigmoid() : activation_base(activation_type::eSigmod)
	{
	}

	virtual void f(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		for (nn_int i = 0; i < len; ++i)
		{
			dst[i] = cOne / (cOne + exp(-src[i]));
		}
	}

	virtual void df(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		for (nn_int i = 0; i < len; ++i)
		{
			nn_float t = cOne / (cOne + exp(-src[i]));
			dst[i] = t * (cOne - t);
		}
	}
};

class activation_relu : public activation_base
{
	nn_float m_leaky;
public:
	activation_relu(nn_float leaky = 0)
		: activation_base(activation_type::eRelu)
		, m_leaky(leaky)
	{
	}

	virtual void f(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		for (nn_int i = 0; i < len; ++i)
		{
			dst[i] = src[i] > 0 ? src[i] : m_leaky * src[i];
		}
	}

	virtual void df(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		for (nn_int i = 0; i < len; ++i)
		{
			dst[i] = src[i] > 0 ? cOne : m_leaky;
		}
	}
};


class activation_softmax : public activation_base
{
public:
	activation_softmax() : activation_base(activation_type::eSoftmax)
	{
	}

	virtual void f(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		nn_float maxv = src[0];
		for (nn_int i = 0; i < len; ++i)
		{
			if (src[i] > maxv)
			{
				maxv = src[i];
			}
		}

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

	virtual void df(const nn_float *nn_restrict src, nn_float *nn_restrict dst, nn_int len)
	{
		nn_assert(false);
	}

};

}

#endif //__ACTIVATION_H__
