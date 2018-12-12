#ifndef __DROPOUT_LAYER_H__
#define __DROPOUT_LAYER_H__

namespace mini_cnn 
{

class dropout_layer : public layer_base
{
protected:
	nn_float m_drop_prob;
	phase_type m_phase_type;
	uniform_random m_uniform_rand;

	struct dropout_task_storage
	{
		std::vector<nn_int> m_drop_mask;
	};
	std::vector<dropout_task_storage> m_dropout_task_storage;

public:
	dropout_layer(nn_float drop_prob)
		: layer_base()
		, m_drop_prob(drop_prob)
		, m_uniform_rand(cZero, cOne)
	{
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);
		m_out_shape = m_prev->m_out_shape;
	}

	virtual void set_phase_type(phase_type phase)
	{
		m_phase_type = phase;
	}

	virtual void set_task_count(nn_int task_count)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		nn_int in_sz = m_prev->m_out_shape.size();
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			if (!m_out_shape.is_img())
			{
				ts.m_x.resize(in_sz);
			}
			else
			{
				ts.m_x.resize(in_w, in_h, in_d);
			}
			ts.m_wd.resize(in_w, in_h, in_d);
		}

		m_dropout_task_storage.resize(task_count);
		for (auto& dts : m_dropout_task_storage)
		{
			dts.m_drop_mask.resize(in_sz);
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		std::vector<nn_int> &drop_mask = m_dropout_task_storage[task_idx].m_drop_mask;
		varray &out_x = m_task_storage[task_idx].m_x;
		nn_int in_sz = input.size();
		if (m_phase_type == phase_type::eTrain)
		{
			for (nn_int i = 0; i < in_sz; ++i)
			{
				if (drop())
				{
					drop_mask[i] = 0;
					out_x[i] = 0;
				}
				else{
					drop_mask[i] = 1;
					out_x[i] = input[i];
				}
			}
		}
		else if (m_phase_type == phase_type::eGradientCheck)
		{
			for (nn_int i = 0; i < in_sz; ++i)
			{
				out_x[i] = drop_mask[i] > 0 ? input[i] : cZero;
			}
		}
		else
		{
			for (nn_int i = 0; i < in_sz; ++i)
			{
				out_x[i] = input[i];
			}
		}

		if (m_next != nullptr)
		{
			m_next->forw_prop(out_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, nn_int task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];
		std::vector<nn_int> &drop_mask = m_dropout_task_storage[task_idx].m_drop_mask;
		nn_int in_sz = drop_mask.size();
		for (nn_int i = 0; i < in_sz; ++i)
		{
			ts.m_wd[i] = drop_mask[i] * next_wd[i];
		}
		m_prev->back_prop(ts.m_wd, task_idx);
	}

	// for gradient check you should fixed the drop probability
	virtual void set_fixed_prop(nn_int task_idx)
	{
		std::vector<nn_int> &drop_mask = m_dropout_task_storage[task_idx].m_drop_mask;
		nn_int sz = (nn_int)drop_mask.size();
		for (nn_int i = 0; i < sz; ++i)
		{
			drop_mask[i] = drop() ? 1 : 0;
		}
	}

private:
	inline bool drop()
	{
		return m_uniform_rand.get_random() <= m_drop_prob;
	}

};
}
#endif //__DROPOUT_LAYER_H__

