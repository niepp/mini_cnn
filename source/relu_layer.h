#ifndef __RELU_H__
#define __RELU_H__

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
protected:
	nn_int m_neural_count;

public:
	relu_layer(nn_int neural_count)
		: layer_base()
	{
		m_neural_count = neural_count;
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

		relu(ts.m_z, ts.m_x);

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

		/*
			prev delta := w * delta กั df(z)
		*/
		deriv_relu(ts.m_z, ts.m_delta);

		const nn_float *nn_restrict vec_next_wd = &next_wd[0];
		nn_float *vec_delta = &ts.m_delta[0];
		for (nn_int i = 0; i < out_sz; ++i)
		{
			vec_delta[i] *= vec_next_wd[i];
		}

		m_prev->back_prop(ts.m_delta, task_idx);

	}

};

}

#endif //__RELU_H__
