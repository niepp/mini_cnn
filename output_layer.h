#ifndef __OUTPUT_LAYER_H__
#define __OUTPUT_LAYER_H__

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>  
using namespace std;

#include "types.h"
#include "utils.h"
#include "math/vectorn.h"
#include "math/matrixmxn.h"
#include "math/mathdef.h"
#include "layer.h"
#include "fully_connected_layer.h"

namespace mini_cnn
{
class OutputLayer : public FullyConnectedLayer
{
protected:
	const VectorN *m_label;  // 学习的正确值
	eLossFunc m_loss_func_type;

	struct TaskStorageOutput
	{
		VectorN *m_label;
	};

	std::vector<TaskStorageOutput> m_task_storage_output;

public:
	OutputLayer(unsigned int neuralCount, eLossFunc lossFunc, eActiveFunc act) : FullyConnectedLayer(neuralCount, act)
	{
		assert(!(lossFunc == eLossFunc::eSigmod_CrossEntropy && act != eActiveFunc::eSigmod)
			&& !(lossFunc == eLossFunc::eSoftMax_LogLikelihood && act != eActiveFunc::eSoftMax)
			);
		m_loss_func_type = lossFunc;
	}

	void SetLabelValue(const VectorN &label)
	{
		m_label = &label;
	}

	void SetLabelValue(const VectorN &label, int task_idx)
	{
		TaskStorageOutput &ts_output = m_task_storage_output[task_idx];
		ts_output.m_label->Copy(label);
	}

	virtual void SetTaskCount(int task_count)
	{
		FullyConnectedLayer::SetTaskCount(task_count);
		m_task_storage_output.resize(task_count);
		for (auto& ts : m_task_storage_output)
		{
			ts.m_label = new VectorN(m_neuralCount);
		}
	}

	virtual void BackProp()
	{
		BackPropImpl(*m_delta, *m_dw, *m_middle, GetInput(), GetOutput(), *m_label);
	}

	virtual void BackProp(int task_idx)
	{
		TaskStorage &ts = m_task_storage[task_idx];
		TaskStorageBase &tsb = GetTaskStorageBase(task_idx);

		VectorN &_d = *ts.m_delta;
		MatrixMN &_w = *ts.m_dw;
		VectorN &_m = *ts.m_middle;
		const VectorN &_in = GetInput(tsb);
		const VectorN &_out = GetOutput(tsb);
		const VectorN &_lab = *m_task_storage_output[task_idx].m_label;
		BackPropImpl(_d, _w, _m, _in, _out, _lab);
	}

	Float GetCost(bool check_gradient)
	{
		return GetCostImpl(check_gradient, GetOutput(), *m_label);
	}

	Float GetCost(bool check_gradient, int task_idx)
	{
		TaskStorageBase &tsb = GetTaskStorageBase(task_idx);
		const VectorN &_out = GetOutput(tsb);
		const VectorN &_lab = *m_task_storage_output[task_idx].m_label;
		return GetCostImpl(check_gradient, _out, _lab);
	}

private:
	void BackPropImpl(VectorN &_d, MatrixMN &_w, VectorN &_m,
		const VectorN &_in, const VectorN &_out, const VectorN &_lab)
	{
		switch (m_loss_func_type)
		{
		case eLossFunc::eMSE:
		{
			m_derived_func(_m);
			_d += (_out - _lab) ^ _m; // 均方误差损失函数对输出层的输出值的偏导数
			_w += _d * _in;
		}
			break;
		case eLossFunc::eSigmod_CrossEntropy:
		{
			// J = -[t * ln(y) + (1 - t) * ln(1 - y)]      : t is the label; y is network output
			// 交叉熵CrossEntropy损失函数和Sigmod激活函数的组合：
			// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
			// ref： http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
			_d += _out - _lab;
			_w += _d * _in;
		}
			break;
		case eLossFunc::eSoftMax_LogLikelihood:
		{
			// J = -t * ln(y)     : t is the label; y is network output
			// LogLikelihood损失函数和Softmax激活函数的组合下：
			// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
			// delta(i) = output(k) - 1    (i==k时， k是one-hot标签对应的index)
			//          = 0                (i!=k时)
			// ref： https://www.cnblogs.com/ZJUT-jiangnan/p/5489047.html
			int idx = _lab.ArgMax();
			Int len = static_cast<Int>(_d.GetSize());
			for (Int i = 0; i < len; ++i)
			{
				_d[i] += (i == idx) ? (_out[i] - (Float)(1.0)) : _out[i];
			}
			_w += _d * _in;
		}
			break;
		default:
			assert(false);
			break;
		}
	}

	Float GetCostImpl(bool check_gradient, const VectorN &_out, const VectorN &_lab)
	{
		Float e = check_gradient ? 0 : cEPSILON;
		Float cost = 0;
		switch (m_loss_func_type)
		{
		case eLossFunc::eMSE:
			{
				VectorN diff = _out - _lab;
				cost = (Float)(0.5) * diff.SquareMagnitude();
			}
			break;
		case eLossFunc::eSigmod_CrossEntropy:
			{
				int len = static_cast<int>(_out.GetSize());
				for (Int i = 0; i < len; ++i)
				{
					Float p = (_lab)[i]; // p is only 0 or 1
					Float q = _out[i];
					Float c = p > 0 ? -log(q + e) : -log((Float)(1.0) - q + e);
					cost += c;
				}
			}
			break;
		case eLossFunc::eSoftMax_LogLikelihood:
			{
				Int idx = _lab.ArgMax();
				cost = -log(_out[idx] + e);
			}
			break;
		default:
			assert(false);
			break;
		}
		return cost;
	}

};
}

#endif //__OUTPUT_LAYER_H__
