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
	void calc_outsize(const shape3d &input, const shape3d &filter, int filter_count, shape3d &output)
	{
		int_t in_w = input.m_w;
		int_t in_h = input.m_h;
		int_t out_w = static_cast<int_t>(::floorf(1.0f * (in_w - filter.m_w) / m_stride_w)) + 1;
		int_t out_h = static_cast<int_t>(::floorf(1.0f * (in_h - filter.m_h) / m_stride_h)) + 1;
		output.set(out_w, out_h, filter_count);
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
		calc_outsize(m_prev->m_out_shape, m_filter_shape, m_filter_count, m_out_shape);
		m_b.resize(m_filter_count);
		m_w.resize(m_filter_shape.m_w, m_filter_shape.m_h, m_filter_shape.m_d, m_filter_count);
	}

	virtual void set_task_count(int_t task_count)
	{
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.resize(m_w.width(), m_w.height(), m_w.depth(), m_w.count());
			ts.m_db.resize(m_filter_count);
			ts.m_z.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
			ts.m_x.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
			ts.m_delta.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
			ts.m_wd.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
		}
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

