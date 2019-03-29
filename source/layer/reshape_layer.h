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
		layer_base::set_task_count(task_count);
		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int out_d = m_out_shape.m_d;
		m_task_storage.resize(task_count);
	}

	virtual void set_batch_size(nn_int batch_size)
	{
		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int out_d = m_out_shape.m_d;
		m_x_vec.resize(out_w, out_h, out_d, batch_size);
		m_wd_vec.resize(m_prev->out_size(), 1, 1, batch_size);
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_assert(input_batch.img_size() == m_out_shape.size());
		nn_assert(input_batch.size() == m_x_vec.size());

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

#endif //__RESHAPE_LAYER_H__
