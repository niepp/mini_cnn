#ifndef __OUTPUT_LAYER_H__
#define __OUTPUT_LAYER_H__

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace mini_cnn
{
class output_layer : public fully_connected_layer
{
protected:
	lossfunc_type m_lossfunc_type;

public:
	output_layer(nn_int neural_count, lossfunc_type lf_type, activation_base *activation)
		: fully_connected_layer(neural_count, activation)
	{
		nn_assert(activation != nullptr);
		nn_assert(!(lf_type == lossfunc_type::eSigmod_CrossEntropy && activation->act_type() != activation_type::eSigmod)
			&& !(lf_type == lossfunc_type::eSoftMax_LogLikelihood && activation->act_type() != activation_type::eSoftmax)
			);
		m_lossfunc_type = lf_type;
	}

	void back_prop(const varray &lab_batch)
	{

		nn_int in_sz = m_w.width();
		nn_int out_sz = m_w.height();

		nn_assert(m_w.check_dim(2));

		nn_int lab_sz = lab_batch.img_size();
		nn_int batch_size = lab_batch.count();

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx)
		{
			layer_base::task_storage &ts = m_task_storage[task_idx];
			const varray &input_batch = m_prev->get_output();

			// loop form batch begin to end
			for (int b = begin; b < end; ++b)
			{
				const nn_float *vec_input = input_batch.data(b);
				const nn_float *vec_lab = lab_batch.data(b);

				calc_delta(vec_lab, lab_sz, task_idx, b);

				/*
					db := delta
					dw := delta * input
				*/
				nn_float *nn_restrict vec_delta = ts.m_delta.data();
				nn_float *nn_restrict vec_db = ts.m_db.data();
				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_db[i] += vec_delta[i];
				}

				fo_vv_m(vec_delta, out_sz
					, vec_input, in_sz
					, ts.m_dw.data());


				/*
					m_w : out_sz X in_sz
					wd := w.transpose * delta
				*/

				fo_mv_v(m_w_t.data(), out_sz, in_sz
					, vec_delta
					, m_wd_vec.data(b));

			}
		});

		m_prev->back_prop(m_wd_vec);

	}

	nn_float calc_cost(bool check_gradient, const varray &lab_batch) const
	{
		const varray &output_batch = get_output();
		nn_int out_sz = output_batch.img_size();
		nn_int batch_size = output_batch.count();
		nn_assert(out_sz == lab_batch.img_size());
		nn_float cost = 0;
		for (nn_int b = 0; b < batch_size; ++b)
		{
			const nn_float *nn_restrict vec_output = output_batch.data(b);
			const nn_float *nn_restrict vec_label = lab_batch.data(b);
			nn_float e = check_gradient ? 0 : cEpsilon;
			switch (m_lossfunc_type)
			{
			case lossfunc_type::eMSE:
				{
					nn_float cur_cost = 0;
					for (nn_int i = 0; i < out_sz; ++i)
					{
						nn_float s = vec_output[i] - vec_label[i];
						cur_cost += s * s;
					}
					cost += cur_cost * (nn_float)(0.5);
				}
				break;
			case lossfunc_type::eSigmod_CrossEntropy:
				{
					for (nn_int i = 0; i < out_sz; ++i)
					{
						nn_float p = vec_label[i]; // p is only 0 or 1
						nn_float q = vec_output[i];
						nn_float c = p > 0 ? -log(q + e) : -log((nn_float)(1.0) - q + e);
						cost += c;
					}
				}
				break;
			case lossfunc_type::eSoftMax_LogLikelihood:
				{
					nn_int idx = arg_max(vec_label, out_sz);
					cost += -log(vec_output[idx] + e);
				}
				break;
			default:
				nn_assert(false);
				break;
			}
		}
		return cost;
	}

private:
	const varray& calc_delta(const nn_float *nn_restrict vec_lab, nn_int lab_sz, nn_int task_idx, nn_int b_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];

		nn_float *nn_restrict vec_delta = ts.m_delta.data();
		const nn_float *nn_restrict vec_x = m_x_vec.data(b_idx);

		switch (m_lossfunc_type)
		{
		case lossfunc_type::eMSE:
			{
				nn_float *nn_restrict vec_z = m_z_vec.data(b_idx);
				m_activation->df(vec_z, vec_delta, lab_sz);
				for (nn_int i = 0; i < lab_sz; ++i)
				{
					vec_delta[i] *= vec_x[i] - vec_lab[i]; // 均方误差损失函数对输出层的输出值的偏导数
				}
			}
			break;
		case lossfunc_type::eSigmod_CrossEntropy:
			{
				// J = -[t * ln(y) + (1 - t) * ln(1 - y)]      : t is the label; y is network output
				// 交叉熵CrossEntropy损失函数和Sigmod激活函数的组合：
				// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
				// ref： http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
				for (nn_int i = 0; i < lab_sz; ++i)
				{
					vec_delta[i] = vec_x[i] - vec_lab[i];
				}
			}
			break;
		case lossfunc_type::eSoftMax_LogLikelihood:
			{
				// J = -t * ln(y)     : t is the label; y is network output
				// LogLikelihood损失函数和Softmax激活函数的组合下：
				// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
				// delta = output - label
				// ref： https://www.cnblogs.com/ZJUT-jiangnan/p/5489047.html
				for (nn_int i = 0; i < lab_sz; ++i)
				{
					vec_delta[i] = vec_x[i] - vec_lab[i];
				}
			}
			break;
		default:
			nn_assert(false);
			break;
		}
		return ts.m_delta;
	}

};
}

#endif //__OUTPUT_LAYER_H__
