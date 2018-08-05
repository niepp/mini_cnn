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
		case eLossFunc::eSoftMax_LogLikelihood:
			// 交叉熵CrossEntropy损失函数和Sigmod激活函数的组合 或者 LogLikelihood损失函数和Softmax激活函数的组合下：
			// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
			// ref： http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
			m_delta->Copy(GetOutput() - *m_label);
			m_dw->Copy(*m_delta * GetInput());
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

	float32_t GetCost()
	{
		float32_t cost = 0;
		switch (m_loss_func_type)
		{
		case eLossFunc::eMSE:
			{
				VectorN diff = GetOutput() - *m_label;
				cost = diff.SquareMagnitude();
			}
			break;
		case eLossFunc::eSigmod_CrossEntropy:
			{
				const VectorN &ov = GetOutput();
				int idx = m_label->ArgMax();
				cost = -ov[idx] * log(ov[idx]);
			}
			break;
		case eLossFunc::eSoftMax_LogLikelihood:
			{
				const VectorN &ov = GetOutput();
				int len = static_cast<int>(ov.GetSize());
				for (int32_t i = 0; i < len; ++i)
				{
					float32_t p = (*m_label)[i]; // p is only 0 or 1
					float32_t q = ov[i];
					float32_t c = p > 0 ? -log(q) : -log(1.0f - q);
					c = std::min(c, 1.0f); // 限定代价值上限，防止数值溢出
					cost += c;
				}
				if (len > 0)
				{
					cost /= len;
				}
			}
			break;
		default:
			assert(false);
			break;
		}
		return cost;
	}

};

#endif //__OUTPUT_LAYER_H__
