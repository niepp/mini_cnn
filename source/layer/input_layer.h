#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__

namespace mini_cnn
{
class input_layer : public layer_base
{
public:
	input_layer(nn_int out_size) : layer_base()
	{
		m_out_shape.set(out_size, 1, 1);
	}

	input_layer(nn_int img_width, nn_int img_height, nn_int img_depth) : layer_base()
	{
		m_out_shape.set(img_width, img_height, img_depth);
	}

	virtual void set_task_count(nn_int task_count)
	{
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			if (m_out_shape.is_img())
			{
				ts.m_x.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
			}
			else
			{
				ts.m_x.resize(m_out_shape.size());
			}
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		nn_assert(m_next != nullptr);

		varray &in = m_task_storage[task_idx].m_x;
		in.copy(input);
		m_next->forw_prop(in, task_idx);
	}

	virtual void back_prop(const varray &next_wd, nn_int task_idx)
	{
	}

};
}
#endif //__INPUT_LAYER_H__
