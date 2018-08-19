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

	virtual void BackProp()
	{
		switch (m_loss_func_type)
		{
		case eLossFunc::eMSE:
			{
				m_prime_func(*m_middle, *m_middle_prime);
				m_delta->Copy(MseDerive() ^ (*m_middle_prime));
				m_dw->Copy(*m_delta * GetInput());
			}
			break;
		case eLossFunc::eSigmod_CrossEntropy:
			{
				// 交叉熵CrossEntropy损失函数和Sigmod激活函数的组合：
				// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
				// ref： http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
				m_delta->Copy(GetOutput() - *m_label);
				m_dw->Copy(*m_delta * GetInput());
			}
			break;
		case eLossFunc::eSoftMax_LogLikelihood:
			{
				// LogLikelihood损失函数和Softmax激活函数的组合下：
				// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
				// delta(i) = output(k) - 1    (i==k时， k是one-hot标签对应的index)
				//          = 0                (i!=k时)
				// ref： https://www.cnblogs.com/ZJUT-jiangnan/p/5489047.html
				m_delta->MakeZero();
				int idx = m_label->ArgMax();
				const VectorN &output = GetOutput();
				(*m_delta)[idx] = output[idx] - (Float)(1.0);
				m_dw->Copy(*m_delta * GetInput());
			}
			break;
		default:
			assert(false);
			break;
		}
	}

	// 均方误差损失函数对输出层的输出值的偏导数
	VectorN MseDerive()
	{
		return GetOutput() - *m_label;
	}

	Float GetCost()
	{
		Float cost = 0;
		switch (m_loss_func_type)
		{
		case eLossFunc::eMSE:
			{
				VectorN diff = GetOutput() - *m_label;
				cost = (Float)(0.5) * diff.SquareMagnitude();
			}
			break;
		case eLossFunc::eSigmod_CrossEntropy:
			{
				const VectorN &ov = GetOutput();
				int len = static_cast<int>(ov.GetSize());
				for (Int i = 0; i < len; ++i)
				{
					Float p = (*m_label)[i]; // p is only 0 or 1
					Float q = ov[i];
					Float c = p > 0 ? -log(q) : -log(1.0f - q);
					//c = std::min(c, 1.0f); // 限定代价值上限，防止数值溢出
					cost += c;
				}
			}
			break;
		case eLossFunc::eSoftMax_LogLikelihood:
			{
				const VectorN &ov = GetOutput();
				int idx = m_label->ArgMax();
				cost = -log(ov[idx]);
				//cost = std::min(cost, 1.0f); // 限定代价值上限，防止数值溢出
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
