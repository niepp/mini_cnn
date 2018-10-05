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
			ts.m_a.resize(out_size());
		}
	}

	virtual const varray& forw_prop(const varray& input, int_t task_idx)
	{
		nn_assert(m_next != nullptr);
		m_task_storage[task_idx].m_a.copy(input);
		return m_next->forw_prop(input, task_idx);
	}

	virtual const varray& back_prop(const varray& next_wd, int_t task_idx)
	{
		return next_wd;
	}

};
}
#endif //__INPUT_LAYER_H__
