#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__

namespace mini_cnn
{

/*
	in the l-th layer:
	 (l)     (l-1)      (l)    (l)
	Z	 =  X       *  W    + B
	 (l)       (l)
	X    =  f(Z   )

	m_w : out_sz X in_sz
	wd := w.transpose * delta
	for relu_layer, w is identity matrix, so wd is equal to ts.m_delta;
	b is zero vector, z of layer l is equal to x
*/

class activation_layer : public layer_base
{
public:
	activation_layer(activation_base *activation)
		: layer_base(activation)
	{
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
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		m_out_shape.set(in_w, in_h, in_d);
	}

	virtual void set_task_count(nn_int task_count)
	{
		layer_base::set_task_count(task_count);
		nn_int in_size = m_prev->m_out_shape.size();

		m_task_storage.resize(task_count);

		for (auto& ts : m_task_storage)
		{
			ts.m_delta.resize(in_size);
		}
	}

	virtual void set_batch_size(nn_int batch_size)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		if (!m_out_shape.is_img())
		{
			m_x_vec.resize(in_w * in_h * in_d, 1, 1, batch_size);
		}
		else
		{
			m_x_vec.resize(in_w, in_h, in_d, batch_size);
		}
		m_wd_vec.resize(in_w, in_h, in_d, batch_size);
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_assert(input_batch.img_size() == m_out_shape.size());
		nn_assert(input_batch.size() == m_x_vec.size());

		nn_int batch_size = input_batch.count();
		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx)
		{
			nn_int img_size = input_batch.img_size();
			for (int b = begin; b < end; ++b)
			{
				const nn_float *input = input_batch.data(b);
				m_activation->f(input, m_x_vec.data(b), img_size);
			}
		});

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}
	}

	virtual void back_prop(const varray &next_wd)
	{
		nn_assert(next_wd.size() == m_x_vec.size());	
		nn_int out_sz = m_x_vec.img_size();
		nn_int batch_size = next_wd.count();

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx)
		{
			layer_base::task_storage &ts = m_task_storage[task_idx];
			nn_assert(ts.m_delta.size() == m_x_vec.img_size());
			for (int b = begin; b < end; ++b)
			{
				const nn_float *vec_next_wd = next_wd.data(b);
				nn_float *vec_wd = m_wd_vec.data(b);
				nn_float *vec_delta = ts.m_delta.data();
				/*
					prev delta := w * delta กั df(z)
					for activation layer the weight matrix is a identity matrix, so wd := w.transpose * delta = delta
				*/
				m_activation->df(m_x_vec.data(b), vec_delta, out_sz);

				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_wd[i] = vec_delta[i] * vec_next_wd[i];
				}
			}
		});

		m_prev->back_prop(m_wd_vec);

	}

};

}

#endif //__ACTIVATION_LAYER_H__
