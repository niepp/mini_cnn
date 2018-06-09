#ifndef __LAYER_H__
#define __LAYER_H__

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

// z = w * a + b
// a = f(z)
class LayerBase
{
public:
	uint32_t m_neuralCount;	
	const VectorN *m_input;
	MatrixMN *m_weight;  // m_weight[i][j] : 前一层的第j个神经元到当前层的第i个神经元的连接权重
	VectorN *m_bias;     // m_bias[i] : 当前层的第i个神经元的偏置
	VectorN *m_middle;	// middle value
	VectorN *m_output;	// output of this layer

	VectorN *m_delta;	// equal to dJ/d(bias)
	MatrixMN *m_dw;		// equal to dJ/d(w)

	VectorN *m_sum_delta;
	MatrixMN *m_sum_dw;

public:
	LayerBase *m_prev, *m_next;

	LayerBase(uint32_t neuralCount) : m_neuralCount(neuralCount), m_prev(NULL), m_next(NULL)
	{
		m_bias = new VectorN(neuralCount);
		m_middle = new VectorN(neuralCount);
		m_output = new VectorN(neuralCount);

	}

	uint32_t Size()
	{
		return m_neuralCount;
	}

	virtual void Connect(LayerBase *next)
	{
		if (next != nullptr)
		{
			next->m_prev = this;
			this->m_next = next;
		}

		if (this->m_prev != nullptr)
		{
			m_weight = new MatrixMN(this->Size(), this->m_prev->Size());

			m_delta = new VectorN(this->Size());
			m_dw = new MatrixMN(this->Size(), this->m_prev->Size());

			m_sum_delta = new VectorN(this->Size());
			m_sum_dw = new MatrixMN(this->Size(), this->m_prev->Size());
		}

		if (next != nullptr)
		{
			next->m_input = this->m_output;
		}
	}

	virtual void Init(NormalRandom nrand)
	{
	}

	virtual void Forward()
	{
	}

	virtual void BackProp()
	{
	}

	virtual void PreTrain()
	{
	}

	virtual void SumGradient()
	{

	}

};

class InputLayer : public LayerBase
{
public:
	InputLayer(uint32_t neuralCount) : LayerBase(neuralCount)
	{
	}
	
	void SetInputData(const VectorN &input)
	{
		m_output->Copy(input);
	}

};

class FullyConnectedLayer : public LayerBase
{
protected:
	ActiveFunc m_activeFunc;

public:
	FullyConnectedLayer(uint32_t neuralCount, ActiveFunc activeFunc) : LayerBase(neuralCount), m_activeFunc(activeFunc)
	{
	}

	virtual void Connect(LayerBase *next)
	{
		LayerBase::Connect(next);
	}

	virtual void Init(NormalRandom nrand)
	{
		for (unsigned int i = 0; i < m_weight->GetRowCount(); ++i)
		{
			for (unsigned int j = 0; j < m_weight->GetColCount(); ++j)
			{
				(*m_weight)(i, j) = nrand.GetRandom();
			}
		}

		for (unsigned int i = 0; i < m_bias->GetSize(); ++i)
		{
			(*m_bias)[i] = nrand.GetRandom();
		}	
	}

	virtual void Forward()
	{
		*m_middle = *m_weight * *m_input + *m_bias;
		*m_output = Sigmoid(*m_middle);
	}

	virtual void BackProp()
	{
		*m_delta = (m_next->m_weight->Transpose() * *m_next->m_delta) ^ SigmoidPrime(*m_middle);
		*m_dw = *m_delta * *m_input;
	}

	virtual void PreTrain()
	{
		m_sum_delta->MakeZero();
		m_sum_dw->MakeZero();
	}

	virtual void SumGradient()
	{
		*m_sum_delta = *m_sum_delta + *m_delta;
		*m_sum_dw = *m_sum_dw + *m_dw;
	}

};

class OutputLayer : public FullyConnectedLayer
{
	const VectorN *m_label;  // 学习的正确值
public:
	OutputLayer(unsigned int neuralCount, ActiveFunc activeFunc) : FullyConnectedLayer(neuralCount, activeFunc)
	{
	}

	void SetLabelValue(const VectorN &label)
	{
		m_label = &label;
	}

	virtual void BackProp()
	{
		*m_delta = LostDerivateAtOutlayer() ^ SigmoidPrime(*m_middle);
		*m_dw = *m_delta * *m_input;
	}

	// 损失函数对输出层的输出值的偏导数 （这里用均方误差函数MSE为例）
	VectorN LostDerivateAtOutlayer()
	{
		return *m_output - *m_label;
	}

};

#endif //__LAYER_H__
