#ifndef __FULLY_CONNECTED_LAYER_H__
#define __FULLY_CONNECTED_LAYER_H__

namespace mini_cnn
{
class fully_connected_layer : public layer_base
{
public:

protected:
	active_func m_f;
	active_func m_df;

public:
	fully_connected_layer(int_t out_size, activation_type ac_type)
		: layer_base(out_size)
	{
		m_b.resize(out_size);
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
		m_w.resize(m_prev->m_out_size, this->m_out_size);
	}

	virtual void set_task_count(int_t task_count)
	{
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.resize(m_w.width(), m_w.height(), m_w.depth());
			ts.m_db.resize(m_b.size());
			ts.m_z.resize(m_b.size());
			ts.m_a.resize(m_b.size());
			ts.m_delta.resize(m_b.size());
			ts.m_wd.resize(m_w.width());
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

		varray &output = m_task_storage[task_idx].m_a;
		for (int_t i = 0; i < height; ++i)
		{
			float_t dot = 0;
			for (int_t j = 0; j < width; ++j)
			{
				dot += m_w(j, i) * input(j);
			}
			ts.m_z(i) = dot + m_b(i);
		}
		m_f(ts.m_z, output);

		if (m_next != nullptr)
		{
			return m_next->forw_prop(output, task_idx);
		}
		return output;
	}

	virtual const varray& back_prop(const varray& next_wd, int_t task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(m_w.dim() == 2);
		nn_assert(next_wd.size() == ts.m_z.size());

		const varray &input = m_prev->output(task_idx);

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

