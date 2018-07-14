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

enum eActiveFunc
{
	eSigmod,
	eRelu,
	eSoftMax,
};

enum eLossFunc
{
	eMSE,
	eSigmod_CrossEntropy,
	eSoftMax_LogLikelihood,
};

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

protected:
	LayerBase *m_prev, *m_next;
	VectorN *m_outputPrime;

public:
	LayerBase(uint32_t neuralCount) : m_neuralCount(neuralCount), m_prev(NULL), m_next(NULL)
	{
		m_bias = new VectorN(neuralCount);
		m_middle = new VectorN(neuralCount);
		m_output = new VectorN(neuralCount);

		m_outputPrime = new VectorN(neuralCount);
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

#endif //__LAYER_H__
