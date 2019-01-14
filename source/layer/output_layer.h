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
	output_layer(nn_int neural_count, lossfunc_type lf_type, activation_type ac_type) : fully_connected_layer(neural_count, ac_type)
	{
		nn_assert(!(lf_type == lossfunc_type::eSigmod_CrossEntropy && ac_type != activation_type::eSigmod)
			&& !(lf_type == lossfunc_type::eSoftMax_LogLikelihood && ac_type != activation_type::eSoftMax)
			);
		m_lossfunc_type = lf_type;
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

	nn_float calc_cost(bool check_gradient, const varray &label, nn_int task_idx) const
	{
		const varray &output = get_output(task_idx);

		const nn_float *nn_restrict vec_output = &output[0];
		const nn_float *nn_restrict vec_label = &label[0];

		nn_int out_sz = output.size();
		nn_assert(out_sz == output.size());

		nn_float e = check_gradient ? 0 : cEpsilon;
		nn_float cost = 0;
		switch (m_lossfunc_type)
		{
			case lossfunc_type::eMSE:
			{
				for (nn_int i = 0; i < out_sz; ++i)
				{
					nn_float s = vec_output[i] - vec_label[i];
					cost += s * s;
				}
				cost *= (nn_float)(0.5);
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
				nn_int idx = label.arg_max();
				cost = -log(vec_output[idx] + e);
			}
			break;
		default:
			nn_assert(false);
			break;
		}
		return cost;
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

		switch (m_lossfunc_type)
		{
		case lossfunc_type::eMSE:
			{
				m_df(ts.m_z, ts.m_delta);
				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_delta[i] *= vec_x[i] - label[i]; // 均方误差损失函数对输出层的输出值的偏导数
				}
			}
			break;
		case lossfunc_type::eSigmod_CrossEntropy:
			{
				// J = -[t * ln(y) + (1 - t) * ln(1 - y)]      : t is the label; y is network output
				// 交叉熵CrossEntropy损失函数和Sigmod激活函数的组合：
				// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
				// ref： http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_delta[i] = vec_x[i] - label[i];
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
				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_delta[i] = vec_x[i] - label[i];
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
