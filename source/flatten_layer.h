#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__

namespace mini_cnn
{

class flatten_layer : public layer_base
{
public:
	flatten_layer()
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
		nn_assert(m_prev->m_out_shape.is_img());

		m_out_shape.set(m_prev->out_size(), 1, 1);

	}

	virtual void set_task_count(nn_int task_count)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;

		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_x.resize(out_size());
			ts.m_wd.resize(in_w, in_h, in_d);
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		nn_assert(input.size() == m_out_shape.size());

		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_assert(input.size() == ts.m_x.size());

		identity(input, ts.m_x);

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

		identity(next_wd, ts.m_wd);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

};

}

#endif //__FLATTEN_LAYER_H__
