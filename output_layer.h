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
#include "fullyconnected_layer.h"


class OutputLayer : public FullyConnectedLayer
{
protected:
	const VectorN *m_label;  // 学习的正确值
	eLossFunc m_lossFuncType;
public:
	OutputLayer(unsigned int neuralCount, eLossFunc lossFunc, eActiveFunc act) : FullyConnectedLayer(neuralCount, act)
	{
		assert(!(lossFunc == eLossFunc::eSigmod_CrossEntropy && act != eActiveFunc::eSigmod)
			&& !(lossFunc == eLossFunc::eSoftMax_LogLikelihood && act != eActiveFunc::eSoftMax)
			);
		m_lossFuncType = lossFunc;
	}

	void SetLabelValue(const VectorN &label)
	{
		m_label = &label;
	}

	virtual void BackProp(LayerBase *next)
	{
		switch (m_lossFuncType)
		{
		case eLossFunc::eMSE:
		{
			m_activePrimeFunc(*m_middle, *m_outputPrime);
			m_delta->Copy(MseDerive() ^ (*m_outputPrime));
			m_dw->Copy(*m_delta * *m_input);
		}
			break;
		case eLossFunc::eSigmod_CrossEntropy:
		case eLossFunc::eSoftMax_LogLikelihood:
			// 交叉熵CrossEntropy损失函数和Sigmod激活函数的组合 或者 LogLikelihood损失函数和Softmax激活函数的组合下：
			// 损失函数对输出层残差的偏导数与激活函数的导数恰好无关
			// ref： http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
			m_delta->Copy(*m_output - *m_label);
			m_dw->Copy(*m_delta * *m_input);
			break;
		default:
			assert(false);
			break;
		}
	}

	// 均方误差损失函数对输出层的输出值的偏导数
	VectorN MseDerive()
	{
		return *m_output - *m_label;
	}

};

#endif //__OUTPUT_LAYER_H__
