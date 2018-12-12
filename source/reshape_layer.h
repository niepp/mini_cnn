#ifndef __RESHAPE_LAYER_H__
#define __RESHAPE_LAYER_H__

namespace mini_cnn
{

class reshape_layer : public layer_base
{
public:
	reshape_layer(nn_int img_width, nn_int img_height, nn_int img_depth)
		: layer_base()
	{
		m_out_shape.set(img_width, img_height, img_depth);
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
		nn_assert(!m_prev->m_out_shape.is_img());
	}

	virtual void set_task_count(nn_int task_count)
	{
		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int out_d = m_out_shape.m_d;
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_x.resize(out_w, out_h, out_d);
			ts.m_wd.resize(m_prev->out_size());
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

#endif //__RESHAPE_LAYER_H__
