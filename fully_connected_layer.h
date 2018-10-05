#ifndef __FULLY_CONNECTED_LAYER_H__
#define __FULLY_CONNECTED_LAYER_H__

namespace mini_cnn
{
class fully_connected_layer : public layer_base
{
protected:
	int_t m_neural_count;
	active_func m_f;
	active_func m_df;

public:
	fully_connected_layer(int_t neural_count, activation_type ac_type)
		: layer_base()
	{
		m_neural_count = neural_count;
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
		m_out_shape.set(m_neural_count, 1, 1);
		m_b.resize(out_size());
		m_w.resize(m_prev->out_size(), out_size());
	}

	virtual void set_task_count(int_t task_count)
	{
		int_t in_sz = m_w.width();
		int_t out_sz = m_w.height(); 
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.resize(in_sz, out_sz);
			ts.m_db.resize(out_sz);
			ts.m_z.resize(out_sz);
			ts.m_x.resize(out_sz);
			ts.m_delta.resize(out_sz);
			ts.m_wd.resize(in_sz);
		}
	}

	virtual const varray& forw_prop(const varray& input, int_t task_idx)
	{
		int_t height = m_w.height();
		int_t width = m_w.width();

		nn_assert(input.dim() == 1);
		nn_assert(width == input.width());

		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(ts.m_z.width() == height);

		for (int_t i = 0; i < height; ++i)
		{
			float_t dot = 0;
			for (int_t j = 0; j < width; ++j)
			{
				dot += m_w(j, i) * input(j);
			}
			ts.m_z(i) = dot + m_b(i);
		}
		m_f(ts.m_z, ts.m_x);

		if (m_next != nullptr)
		{
			return m_next->forw_prop(ts.m_x, task_idx);
		}
		return ts.m_x;
	}

	virtual const varray& back_prop(const varray& next_wd, int_t task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(m_w.dim() == 2);
		nn_assert(next_wd.size() == ts.m_z.size());

		const varray &input = m_prev->get_output(task_idx);

		int_t out_sz = next_wd.size();
		int_t in_sz = input.size();

		nn_assert(in_sz == m_w.width());
		nn_assert(out_sz == m_w.height());

		m_df(ts.m_z, ts.m_delta);
		for (int_t i = 0; i < out_sz; ++i)
		{
			ts.m_delta(i) *= next_wd(i);
		}

		/*
			dw = db * input
		*/
		for (int_t i = 0; i < out_sz; ++i)
		{
			ts.m_db(i) += ts.m_delta(i);
			for (int_t j = 0; j < in_sz; ++j)
			{
				ts.m_dw(j, i) += ts.m_delta(i) * input(j);
			}
		}

		/*
			m_w : out_sz X in_sz
		*/
		for (int_t i = 0; i < in_sz; ++i)
		{
			float_t dot = 0;
			for (int_t j = 0; j < out_sz; ++j)
			{
				dot += m_w(i, j) * ts.m_delta(j);
			}
			ts.m_wd(i) = dot;
		}

		return m_prev->back_prop(ts.m_wd, task_idx);

	}

};
}
#endif //__FULLY_CONNECTED_LAYER_H__

