#ifndef __FULLY_CONNECTED_LAYER_H__
#define __FULLY_CONNECTED_LAYER_H__

namespace mini_cnn
{
class fully_connected_layer : public layer_base
{
protected:
	nn_int m_neural_count;
	active_func m_f;
	active_func m_df;
	varray m_tw;  // transpose of m_w

public:
	fully_connected_layer(nn_int neural_count, activation_type ac_type)
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

	virtual nn_int fan_in_size() const
	{
		return m_prev->out_size();
	}

	virtual nn_int fan_out_size() const
	{
		return out_size();
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);
		m_out_shape.set(m_neural_count, 1, 1);
		m_b.resize(out_size());
		m_w.resize(m_prev->out_size(), out_size());
		m_tw.resize(out_size(), m_prev->out_size());
	}

	virtual void set_task_count(nn_int task_count)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;

		nn_int in_sz = m_w.width();
		nn_int out_sz = m_w.height(); 
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.resize(in_sz, out_sz);
			ts.m_db.resize(out_sz);
			ts.m_z.resize(out_sz);
			ts.m_x.resize(out_sz);
			ts.m_delta.resize(out_sz);
			if (m_prev->m_out_shape.is_img())
			{
				ts.m_wd.resize(in_w, in_h, in_d);
			}
			else
			{
				ts.m_wd.resize(in_sz);
			}
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		nn_int height = m_w.height();
		nn_int width = m_w.width();

		nn_assert(width == input.size());

		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(ts.m_z.width() == height);

		for (nn_int i = 0; i < height; ++i)
		{
			nn_float dot = vec_dot(&m_w(0, i), &input[0], width);
			ts.m_z(i) = dot + m_b(i);
		}
		m_f(ts.m_z, ts.m_x);

		if (m_next != nullptr)
		{
			m_next->forw_prop(ts.m_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, nn_int task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(m_w.dim() == 2);
		nn_assert(next_wd.size() == ts.m_z.size());

		const varray &input = m_prev->get_output(task_idx);

		nn_int out_sz = next_wd.size();
		nn_int in_sz = input.size();

		nn_assert(in_sz == m_w.width());
		nn_assert(out_sz == m_w.height());

		m_df(ts.m_z, ts.m_delta);

		const nn_float *nn_restrict vec_next_wd = &next_wd[0];
		nn_float *nn_restrict vec_delta = &ts.m_delta[0];
		for (nn_int i = 0; i < out_sz; ++i)
		{
			vec_delta[i] *= vec_next_wd[i];
		}

		/*
			dw = db * input
		*/
		nn_float *nn_restrict vec_db = &ts.m_db[0];
		for (nn_int i = 0; i < out_sz; ++i)
		{
			vec_db[i] += vec_delta[i];
		}
		for (nn_int i = 0; i < out_sz; ++i)
		{
			nn_float *nn_restrict vec_dw_i = &ts.m_dw(0, i);
			const nn_float *nn_restrict vec_input = &input[0];
			for (nn_int j = 0; j < in_sz; ++j)
			{
				vec_dw_i[j] = vec_delta[i] * vec_input[j];
			}
		}

		/*
			m_w : out_sz X in_sz
			wd = w * delta
		*/
		for (nn_int i = 0; i < in_sz; ++i)
		{
			ts.m_wd[i] = vec_dot(&m_tw(0, i), vec_delta, out_sz);
		}

		m_prev->back_prop(ts.m_wd, task_idx);

	}

	virtual void update_weights(nn_float eff)
	{
		layer_base::update_weights(eff);

		nn_int out_sz = m_tw.width();
		nn_int in_sz = m_tw.height();
		for (nn_int i = 0; i < out_sz; ++i)
		{
			for (nn_int j = 0; j < in_sz; ++j)
			{
				m_tw(i, j) = m_w(j, i);
			}
		}
	}

private:
	static inline nn_float vec_dot(const nn_float *nn_restrict v1, const nn_float *nn_restrict v2, nn_int len)
	{
		nn_float s = 0;
		for (nn_int i = 0; i < len; ++i)
		{
			s += v1[i] * v2[i];
		}
		return s;
	}

};
}
#endif //__FULLY_CONNECTED_LAYER_H__

