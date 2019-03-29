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
		layer_base::set_task_count(task_count);
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		m_task_storage.resize(task_count);
	}

	virtual void set_batch_size(nn_int batch_size)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		m_x_vec.resize(out_size(), 1, 1, batch_size);
		m_wd_vec.resize(in_w, in_h, in_d, batch_size);
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_int batch_size = input_batch.count();
		nn_int img_size = input_batch.img_size();
		nn_assert(img_size == m_out_shape.size());

		m_x_vec.copy(input_batch);

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}
	}

	virtual void back_prop(const varray &next_wd)
	{
		nn_int batch_size = next_wd.count();
		nn_assert(next_wd.size() == m_x_vec.size());

		m_wd_vec.copy(next_wd);

		m_prev->back_prop(m_wd_vec);

	}

};

}

#endif //__FLATTEN_LAYER_H__
