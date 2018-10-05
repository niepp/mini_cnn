#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

namespace mini_cnn 
{

enum class padding
{
	valid, // only use valid pixels of input
	same   // padding zero around input to keep image size
};

/*
for the l-th Conv layer:
	(l)     (l-1)      (l)    (l)
   Z	=  X       *  W    + B
    (l)      (l)
   X    = f(Z   )
*/
class convolutional_layer : public layer_base
{

protected:
	shape3d m_filter_shape;
	int_t m_filter_count;
	int_t m_stride_w;
	int_t m_stride_h;
	padding m_padding;
	active_func m_f;
	active_func m_df;
protected:
	int_t calc_outsize() const
	{
		return 0;
	}
public:
	convolutional_layer(int_t filter_w, int_t filter_h, int_t filter_c, int_t filter_n, int_t stride_w, int_t stride_h, padding padding, activation_type ac_type)
		: layer_base()
		, m_filter_shape(filter_w, filter_h, filter_c)
		, m_stride_w(stride_w), m_stride_h(stride_h), m_padding(padding)
	{
		switch (ac_type)
		{
		case activation_type::eSigmod:
			m_f = sigmoid;
			m_df = deriv_sigmoid;
			break;
		case activation_type::eRelu:
			m_f = relu;
			m_df = deriv_relu;
			break;
		case activation_type::eSoftMax:
			m_f = softmax;
			m_df = nullptr;
			break;
		default:
			break;
		}
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);

		int_t in_w = m_prev->m_out_shape.m_w;
		int_t in_h = m_prev->m_out_shape.m_h;
		int_t out_w = static_cast<int_t>(::floorf(1.0f * (in_w - m_filter_shape.m_w) / m_stride_w)) + 1;
		int_t out_h = static_cast<int_t>(::floorf(1.0f * (in_h - m_filter_shape.m_h) / m_stride_h)) + 1;
		m_out_shape.set(out_w, out_h, m_filter_count);

		m_b.resize(m_filter_count);
		m_w.resize(m_filter_shape.m_w, m_filter_shape.m_h, m_filter_shape.m_d, m_filter_count);
	}

	virtual void set_task_count(int_t task_count)
	{
	}

	virtual const varray& forw_prop(const varray& input, int_t task_idx)
	{
		return m_next->forw_prop(input, task_idx);
	}

	virtual const varray& back_prop(const varray& next_wd, int_t task_idx)
	{
		return next_wd;
	}

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

