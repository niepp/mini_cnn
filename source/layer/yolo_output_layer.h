#ifndef __YOLO_OUTPUT_LAYER_H__
#define __YOLO_OUTPUT_LAYER_H__

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace mini_cnn
{
class yolo_output_layer : public fully_connected_layer
{
protected:

public:
	yolo_output_layer(nn_int neural_count, activation_type ac_type) : fully_connected_layer(neural_count, ac_type)
	{
	}

	void backward(const varray &label, nn_int task_idx)
	{
		calc_delta(label, task_idx);

		layer_base::task_storage &ts = m_task_storage[task_idx];
		const varray &input = m_prev->get_output(task_idx);
		nn_int out_sz = get_output(task_idx).size();
		nn_int in_sz = input.size();

		nn_assert(m_w.check_dim(2));
		nn_assert(in_sz == m_w.width());
		nn_assert(out_sz == m_w.height());

		/*
		db := delta
		dw := delta * input
		*/
		nn_float *nn_restrict vec_delta = &ts.m_delta[0];
		nn_float *nn_restrict vec_db = &ts.m_db[0];
		for (nn_int i = 0; i < out_sz; ++i)
		{
			vec_db[i] = vec_delta[i];
		}

		fo_vv_m(vec_delta, out_sz
			, &input[0], in_sz
			, &ts.m_dw[0]);

		/*
		m_w : out_sz X in_sz
		wd := w.transpose * delta
		*/
		fo_mv_v(&m_w_t[0], out_sz, in_sz
			, vec_delta
			, &ts.m_wd[0]);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	const varray& calc_delta(const varray &label, nn_int task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_int out_sz = label.size();
		nn_assert(out_sz == ts.m_z.size());

		nn_float *nn_restrict vec_delta = &ts.m_delta[0];
		const nn_float *nn_restrict vec_x = &ts.m_x[0];
		const nn_float *nn_restrict vec_label = &label[0];
		m_df(ts.m_z, ts.m_delta);
		for (nn_int i = 0; i < out_sz; ++i)
		{
			vec_delta[i] *= vec_x[i] - label[i]; // 均方误差损失函数对输出层的输出值的偏导数
		}
		return ts.m_delta;
	}

};
}

#endif //__YOLO_OUTPUT_LAYER_H__
