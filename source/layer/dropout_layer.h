#ifndef __DROPOUT_LAYER_H__
#define __DROPOUT_LAYER_H__

namespace mini_cnn 
{

class dropout_layer : public layer_base
{
protected:
	nn_float m_drop_prob;
	uniform_random m_uniform_rand;

	struct dropout_task_storage
	{
		std::vector<index_vec> m_drop_mask_vec;
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

	virtual void set_task_count(nn_int task_count)
	{
		layer_base::set_task_count(task_count);
		m_task_storage.resize(task_count);
		m_dropout_task_storage.resize(task_count);

	}

	virtual void set_batch_size(nn_int batch_size)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		nn_int in_sz = m_prev->m_out_shape.size();

		if (!m_out_shape.is_img())
		{
			m_x_vec.resize(in_sz, 1, 1, batch_size);
		}
		else
		{
			m_x_vec.resize(in_w, in_h, in_d, batch_size);
		}
		m_wd_vec.resize(in_w, in_h, in_d, batch_size);

		for (auto& dts : m_dropout_task_storage)
		{
			dts.m_drop_mask_vec.resize(batch_size);
			for (auto &mask : dts.m_drop_mask_vec)
			{
				mask.resize(in_sz);
			}
		}

	}

	virtual void forw_prop(const varray &input_batch)
	{
		std::vector<index_vec> &drop_mask_vec = m_dropout_task_storage[0].m_drop_mask_vec;

		nn_int batch_size = input_batch.count();
		nn_int in_sz = input_batch.img_size();

		if (m_phase_type == phase_type::eTrain)
		{
			for (nn_int b = 0; b < batch_size; ++b)
			{
				const nn_float *input = input_batch.data(b);
				nn_float *out_x = m_x_vec.data(b);
				index_vec &drop_mask = drop_mask_vec[b];
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
		}
		else if (m_phase_type == phase_type::eGradientCheck)
		{
			for (nn_int b = 0; b < batch_size; ++b)
			{
				const nn_float *input = input_batch.data(b);
				nn_float *out_x = m_x_vec.data(b);
				index_vec &drop_mask = drop_mask_vec[b];
				for (nn_int i = 0; i < in_sz; ++i)
				{
					out_x[i] = drop_mask[i] > 0 ? input[i] : cZero;
				}
			}
		}
		else
		{
			for (nn_int b = 0; b < batch_size; ++b)
			{
				const nn_float *input = input_batch.data(b);
				nn_float *out_x = m_x_vec.data(b);
				for (nn_int i = 0; i < in_sz; ++i)
				{
					out_x[i] = input[i];
				}
			}
		}

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}
	}

	virtual void back_prop(const varray &next_wd)
	{
		std::vector<index_vec> &drop_mask_vec = m_dropout_task_storage[0].m_drop_mask_vec;

		nn_int batch_size = next_wd.count();
		for (nn_int b = 0; b < batch_size; ++b)
		{
			index_vec &drop_mask = drop_mask_vec[b];
			nn_int in_sz = static_cast<nn_int>(drop_mask.size());

			const nn_float *nn_restrict vec_next_wd = next_wd.data(b);
			nn_float *nn_restrict vec_wd = m_wd_vec.data(b);
			for (nn_int i = 0; i < in_sz; ++i)
			{
				vec_wd[i] = drop_mask[i] * vec_next_wd[i];
			}
		}
		m_prev->back_prop(m_wd_vec);
	}

	// for gradient check you should fixed the drop probability
	virtual void set_fixed_prop(nn_int task_idx)
	{
		std::vector<index_vec> &drop_mask_vec = m_dropout_task_storage[0].m_drop_mask_vec;
		index_vec &drop_mask = drop_mask_vec[0];
		nn_int in_sz = static_cast<nn_int>(drop_mask.size());
		for (nn_int i = 0; i < in_sz; ++i)
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

