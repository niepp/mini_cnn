#ifndef __RELU_LAYER_H__
#define __RELU_LAYER_H__

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

class relu_layer : public layer_base
{
public:
	relu_layer()
		: layer_base()
	{
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
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		nn_int in_size = m_prev->m_out_shape.size();

		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			if (!m_out_shape.is_img())
			{
				ts.m_x.resize(in_w * in_h * in_d);
			}
			else
			{
				ts.m_x.resize(in_w, in_h, in_d);
			}
			ts.m_delta.resize(in_size);
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		nn_assert(input.size() == m_out_shape.size());

		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(input.size() == ts.m_x.size());

		relu(input, ts.m_x);

		if (m_next != nullptr)
		{
			m_next->forw_prop(ts.m_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, nn_int task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(next_wd.size() == ts.m_x.size());

		nn_int out_sz = next_wd.size();

		/*
			prev delta := w * delta กั df(z)
		*/
		deriv_relu(ts.m_x, ts.m_delta);

		const nn_float *nn_restrict vec_next_wd = &next_wd[0];
		nn_float *nn_restrict vec_delta = &ts.m_delta[0];
		for (nn_int i = 0; i < out_sz; ++i)
		{
			vec_delta[i] *= vec_next_wd[i];
		}

		m_prev->back_prop(ts.m_delta, task_idx);

	}

};

}

#endif //__RELU_LAYER_H__
