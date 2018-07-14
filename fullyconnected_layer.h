#ifndef __FULLYCONNECTED_LAYER_H__
#define __FULLYCONNECTED_LAYER_H__

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

class FullyConnectedLayer : public LayerBase
{
protected:
	eActiveFunc m_activeFuncType;
	ActiveFunc m_activeFunc;
	ActiveFunc m_activePrimeFunc;

public:
	FullyConnectedLayer(uint32_t neuralCount, eActiveFunc act) : LayerBase(neuralCount)
	{
		m_activeFuncType = act;
		switch (act)
		{
		case eActiveFunc::eSigmod:
			m_activeFunc = Sigmoid;
			m_activePrimeFunc = SigmoidPrime;
			break;
		case eActiveFunc::eRelu:
			m_activeFunc = Relu;
			m_activePrimeFunc = ReluPrime;
			break;
		case eActiveFunc::eSoftMax:
			m_activeFunc = Softmax;
			break;
		default:
			break;
		}
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
		m_middle->Copy(*m_weight * *m_input + *m_bias);
		m_activeFunc(*m_middle, *m_output);
	}

	virtual void BackProp()
	{
		m_activePrimeFunc(*m_middle, *m_outputPrime);
		m_delta->Copy((m_next->m_weight->Transpose() * (*m_next->m_delta)) ^ (*m_outputPrime));
		m_dw->Copy(*m_delta * *m_input);
	}

	virtual void PreTrain()
	{
		m_sum_delta->MakeZero();
		m_sum_dw->MakeZero();
	}

	virtual void SumGradient()
	{
		m_sum_delta->Copy(*m_sum_delta + *m_delta);
		m_sum_dw->Copy(*m_sum_dw + *m_dw);
	}

};

#endif //__FULLYCONNECTED_LAYER_H__
