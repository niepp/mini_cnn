#ifndef __WEIGHT_INITIALIZER_H__
#define __WEIGHT_INITIALIZER_H__

namespace mini_cnn
{

class weight_initializer
{
public:
	weight_initializer()
	{
	}
	virtual void operator()(std::vector<layer_base*> &layers) = 0;
};


/* 
	[Gaussian initialize]
	Weights are randomly drawn from Gaussian distributions with fixed mean(e.g., 0) and fixed standard deviation(e.g., 0.01).
	This is the most common initialization method in deep learning.
*/
class truncated_normal_initializer : public weight_initializer
{
	float_t m_bias_constant;
	normal_random m_normal_random;
public:
	truncated_normal_initializer(std::mt19937_64 generator, float_t mean = 0, float_t stdev = 1.0, int_t truncated = 3, float_t bias_constant = 0.1)
		: m_normal_random(generator, mean, stdev, truncated), m_bias_constant(bias_constant)
	{
	}

	virtual void operator()(std::vector<layer_base*> &layers)
	{
		for (auto &layer : layers)
		{
			varray &w = layer->m_w;
			varray &b = layer->m_b;

			int_t w_sz = w.size();
			for (int_t i = 0; i < w_sz; ++i)
			{
				w[i] = m_normal_random.get_random();
			}

			int_t b_sz = b.size();
			for (int_t i = 0; i < b_sz; ++i)
			{
				b[i] = m_bias_constant;
			}
		}
	}
};


/*
	[Xavier initialize]
	This initializer is designed to keep the scale of the gradients roughly the same in all layers.
	In uniform distribution this ends up being the range : x = sqrt(6. / (in + out)); [-x, x] and for normal distribution a standard deviation of sqrt(2. / (in + out)) is used.

	ref: Xavier Glorot and Yoshua Bengio(2010) : Understanding the difficulty of training deep feedforward neural networks.International conference on artificial intelligence and statistics.
*/
class xavier_normal_initializer : public weight_initializer
{
	int_t m_truncated;
	float_t m_bias_constant;
	std::mt19937_64 m_generator;
public:
	xavier_normal_initializer(std::mt19937_64 generator, int_t truncated = 3, float_t bias_constant = 0.1)
		: m_generator(generator), m_truncated(truncated), m_bias_constant(bias_constant)
	{
	}

	virtual void operator()(std::vector<layer_base*> &layers)
	{
		for (auto &layer : layers)
		{
			varray &w = layer->m_w;
			varray &b = layer->m_b;

			float_t stdev = sqrt(2.0 / (layer->fan_in_size() + layer->fan_out_size()));
			normal_random nrand(m_generator, 0, stdev, m_truncated);

			int_t w_sz = w.size();
			for (int_t i = 0; i < w_sz; ++i)
			{
				w[i] = nrand.get_random();
			}

			int_t b_sz = b.size();
			for (int_t i = 0; i < b_sz; ++i)
			{
				b[i] = m_bias_constant;
			}
		}
	}
};

class xavier_uniform_initializer : public weight_initializer
{
	float_t m_bias_constant;
	std::mt19937_64 m_generator;
public:
	xavier_uniform_initializer(std::mt19937_64 generator, float_t bias_constant = 0.1)
		: m_generator(generator), m_bias_constant(bias_constant)
	{
	}

	virtual void operator()(std::vector<layer_base*> &layers)
	{
		for (auto &layer : layers)
		{
			varray &w = layer->m_w;
			varray &b = layer->m_b;

			float_t range = std::sqrt(6.0 / (layer->fan_in_size() + layer->fan_out_size()));
			uniform_random urand(m_generator, -range, range);

			int_t w_sz = w.size();
			for (int_t i = 0; i < w_sz; ++i)
			{
				w[i] = urand.get_random();
			}

			int_t b_sz = b.size();
			for (int_t i = 0; i < b_sz; ++i)
			{
				b[i] = m_bias_constant;
			}
		}
	}
};


/*
	[He initialize]
	Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Delving Deep into Rectifiers : Surpassing Human - Level Performance on ImageNet Classification, Technical report, arXiv, Feb. 2015
*/
class he_normal_initializer : public weight_initializer
{
	int_t m_truncated;
	float_t m_bias_constant;
	std::mt19937_64 m_generator;
public:
	he_normal_initializer(std::mt19937_64 generator, int_t truncated = 3, float_t bias_constant = 0.1)
		: m_generator(generator), m_truncated(truncated), m_bias_constant(bias_constant)
	{
	}

	virtual void operator()(std::vector<layer_base*> &layers)
	{
		for (auto &layer : layers)
		{
			varray &w = layer->m_w;
			varray &b = layer->m_b;

			float_t stdev = sqrt(2.0 / (layer->fan_in_size()));
			normal_random nrand(m_generator, 0, stdev, m_truncated);

			int_t w_sz = w.size();
			for (int_t i = 0; i < w_sz; ++i)
			{
				w[i] = nrand.get_random();
			}

			int_t b_sz = b.size();
			for (int_t i = 0; i < b_sz; ++i)
			{
				b[i] = m_bias_constant;
			}
		}
	}
};

class he_uniform_initializer : public weight_initializer
{
	float_t m_bias_constant;
	std::mt19937_64 m_generator;
public:
	he_uniform_initializer(std::mt19937_64 generator, float_t bias_constant = 0.1)
		: m_generator(generator), m_bias_constant(bias_constant)
	{
	}

	virtual void operator()(std::vector<layer_base*> &layers)
	{
		for (auto &layer : layers)
		{
			varray &w = layer->m_w;
			varray &b = layer->m_b;

			float_t range = std::sqrt(6.0 / (layer->fan_in_size()));
			uniform_random urand(m_generator, -range, range);

			int_t w_sz = w.size();
			for (int_t i = 0; i < w_sz; ++i)
			{
				w[i] = urand.get_random();
			}

			int_t b_sz = b.size();
			for (int_t i = 0; i < b_sz; ++i)
			{
				b[i] = m_bias_constant;
			}
		}
	}
};

}
#endif //__WEIGHT_INITIALIZER_H__
