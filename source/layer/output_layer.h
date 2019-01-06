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
			dw = db * input
		*/
		for (nn_int i = 0; i < out_sz; ++i)
		{
			ts.m_db(i) += ts.m_delta(i);
			for (nn_int j = 0; j < in_sz; ++j)
			{
				ts.m_dw(j, i) += ts.m_delta(i) * input[j];
			}
		}

		/*
			m_w : out_sz X in_sz
		*/
		for (nn_int i = 0; i < in_sz; ++i)
		{
			nn_float dot = 0;
			for (nn_int j = 0; j < out_sz; ++j)
			{
				dot += m_w(i, j) * ts.m_delta(j);
			}
			ts.m_wd[i] = dot;
		}

		m_prev->back_prop(ts.m_wd, task_idx);

	}

	nn_float calc_cost(bool check_gradient, const varray &label, nn_int task_idx) const
	{
		const varray &output = get_output(task_idx);

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
					nn_float s = output(i) - label(i);
					cost += s * s;
				}
				cost *= (nn_float)(0.5);
			}
			break;
		case lossfunc_type::eSigmod_CrossEntropy:
			{
				for (nn_int i = 0; i < out_sz; ++i)
				{
					nn_float p = label(i); // p is only 0 or 1
					nn_float q = output(i);
					nn_float c = p > 0 ? -log(q + e) : -log((nn_float)(1.0) - q + e);
					cost += c;
				}
			}
			break;
		case lossfunc_type::eSoftMax_LogLikelihood:
			{
				nn_int idx = label.arg_max();
				cost = -log(output(idx) + e);
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

		switch (m_lossfunc_type)
		{
		case lossfunc_type::eMSE:
			{
				m_df(ts.m_z, ts.m_delta);
				for (nn_int i = 0; i < out_sz; ++i)
				{
					ts.m_delta(i) *= ts.m_x(i) - label(i); // ���������ʧ���������������ֵ��ƫ����
				}
			}
			break;
		case lossfunc_type::eSigmod_CrossEntropy:
			{
				// J = -[t * ln(y) + (1 - t) * ln(1 - y)]      : t is the label; y is network output
				// ������CrossEntropy��ʧ������Sigmod���������ϣ�
				// ��ʧ�����������в��ƫ�����뼤����ĵ���ǡ���޹�
				// ref�� http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
				for (nn_int i = 0; i < out_sz; ++i)
				{
					ts.m_delta(i) = ts.m_x(i) - label(i);
				}
			}
			break;
		case lossfunc_type::eSoftMax_LogLikelihood:
			{
				// J = -t * ln(y)     : t is the label; y is network output
				// LogLikelihood��ʧ������Softmax�����������£�
				// ��ʧ�����������в��ƫ�����뼤����ĵ���ǡ���޹�
				// delta = output - label
				// ref�� https://www.cnblogs.com/ZJUT-jiangnan/p/5489047.html
				for (nn_int i = 0; i < out_sz; ++i)
				{
					ts.m_delta(i) = ts.m_x(i) - label(i);
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
