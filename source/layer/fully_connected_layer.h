#ifndef __FULLY_CONNECTED_LAYER_H__
#define __FULLY_CONNECTED_LAYER_H__

namespace mini_cnn
{
class fully_connected_layer : public layer_base
{
protected:
	nn_int m_neural_count;
	varray m_w_t;		 // weight matrix(m_w)'s transpose

public:
	fully_connected_layer(nn_int neural_count, activation_base *activation)
		: layer_base(activation)
	{
		m_neural_count = neural_count;
		nn_assert(activation != nullptr);
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
		m_w_t.resize(out_size(), m_prev->out_size());
	}

	virtual void set_task_count(nn_int task_count)
	{
		layer_base::set_task_count(task_count);
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
			ts.m_delta.resize(out_sz);
		}
		// weight matrix have been initialized
		transpose(m_w, m_w_t);
	}

	virtual void set_batch_size(nn_int batch_size)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		nn_int in_sz = m_w.width();
		nn_int out_sz = m_w.height();
		m_z_vec.resize(out_sz, 1, 1, batch_size);
		m_x_vec.resize(out_sz, 1, 1, batch_size);
		if (m_prev->m_out_shape.is_img())
		{
			m_wd_vec.resize(in_w, in_h, in_d, batch_size);
		}
		else
		{
			m_wd_vec.resize(in_sz, 1, 1, batch_size);
		}
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_int height = m_w.height();
		nn_int width = m_w.width();
		nn_int batch_size = input_batch.count();
		nn_int img_size = input_batch.img_size();
		nn_assert(img_size == width);

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx)
		{
			for (int b = begin; b < end; ++b)
			{
				const nn_float *input = input_batch.data(b);
				nn_float *vec_z = m_z_vec.data(b);

				// z = w * input + b
				fo_mv_v(m_w.data(), width, height
					, input
					, vec_z);

				nn_float *vec_b = m_b.data();
				for (nn_int i = 0; i < height; ++i)
				{
					vec_z[i] += vec_b[i];
				}

				nn_float *nn_restrict output = &m_x_vec(0, 0, 0, b);
				m_activation->f(vec_z, output, height);
			}
		});

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}

	}

	virtual void back_prop(const varray &next_wd)
	{
		nn_int in_sz = m_w.width();
		nn_int out_sz = m_w.height();
		nn_int batch_size = next_wd.count();

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx)
		{
			layer_base::task_storage &ts = m_task_storage[task_idx];
			const varray &input_batch = m_prev->get_output();

			for (int b = begin; b < end; ++b)
			{
				const nn_float *vec_next_wd = next_wd.data(b);
				nn_float *vec_z = m_z_vec.data(b);
				nn_float *vec_delta = ts.m_delta.data();
				/*
					prev delta := w * delta กั df(z)
				*/
				m_activation->df(vec_z, vec_delta, ts.m_delta.size());

				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_delta[i] *= vec_next_wd[i];
				}

				/*
					db := delta
					dw := delta * input
				*/
				nn_float *vec_db = ts.m_db.data();
				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_db[i] += vec_delta[i];
				}

				const nn_float *vec_input = input_batch.data(b);

				fo_vv_m(vec_delta, out_sz
					, vec_input, in_sz
					, ts.m_dw.data());

				/*
					m_w : out_sz X in_sz
					wd := w.transpose * delta
				*/
				fo_mv_v(m_w_t.data(), out_sz, in_sz
					, vec_delta
					, m_wd_vec.data(b));

			}
		});

		m_prev->back_prop(m_wd_vec);

	}

	virtual bool update_weights(nn_float batch_lr)
	{
		bool succ = layer_base::update_weights(batch_lr);
		transpose(m_w, m_w_t);
		return succ;
	}

};
}
#endif //__FULLY_CONNECTED_LAYER_H__

