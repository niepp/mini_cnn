#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__

namespace mini_cnn
{
class input_layer : public layer_base
{
public:
	input_layer(int_t out_size) : layer_base()
	{
		m_out_shape.set(out_size, 1, 1);
	}

	input_layer(int_t img_width, int_t img_height, int_t img_depth) : layer_base()
	{
		m_out_shape.set(img_width, img_height, img_depth);
	}

	virtual void set_task_count(int_t task_count)
	{
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_x.resize(out_size());
		}
	}

	virtual void forw_prop(const varray &input, int_t task_idx)
	{
		nn_assert(m_next != nullptr);

		varray &in = m_task_storage[task_idx].m_x;
		in.copy(input);
		if (m_out_shape.is_img())
		{
			in.reshape(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
		}
		m_next->forw_prop(in, task_idx);
	}

	virtual void back_prop(const varray &next_wd, int_t task_idx)
	{
	}

};
}
#endif //__INPUT_LAYER_H__
