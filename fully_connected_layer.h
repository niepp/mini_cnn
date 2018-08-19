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

namespace mini_cnn
{
class FullyConnectedLayer : public LayerBase
{
public:
	MatrixMN *m_weight;  // m_weight[i][j] : 前一层的第j个神经元到当前层的第i个神经元的连接权重
	VectorN *m_bias;     // m_bias[i] : 当前层的第i个神经元的偏置
	VectorN *m_middle;	// middle value

	//const VectorN *m_input;		// input of this layer, this is a ref to prev layer's output
	//VectorN *m_output;			// output of this layer
	VectorN *m_middle_prime;

	VectorN *m_delta;	// equal to dJ/d(bias)
	MatrixMN *m_dw;		// equal to dJ/d(w)

	VectorN *m_sum_delta;
	MatrixMN *m_sum_dw;

protected:
	eActiveFunc m_func_type;
	ActiveFunc m_func;
	ActiveFunc m_prime_func;

public:
	FullyConnectedLayer(uInt neuralCount, eActiveFunc act)
		: LayerBase(neuralCount, new VectorInOut(), new VectorInOut())
	{
		m_bias = new VectorN(neuralCount);
		m_middle = new VectorN(neuralCount);
	
		//VectorInOut* vec_in = dynamic_cast<VectorInOut*>(m_input);
		//vec_in->m_value = new VectorN(neuralCount);

		VectorInOut* vec_out = dynamic_cast<VectorInOut*>(m_output);
		vec_out->m_value = new VectorN(neuralCount);

		m_middle_prime = new VectorN(neuralCount);

		m_func_type = act;
		switch (act)
		{
		case eActiveFunc::eSigmod:
			m_func = Sigmoid;
			m_prime_func = SigmoidPrime;
			break;
		case eActiveFunc::eTanh:
			m_func = Tanh;
			m_prime_func = TanhPrime;
			break;
		case eActiveFunc::eRelu:
			m_func = Relu;
			m_prime_func = ReluPrime;
			break;
		case eActiveFunc::eSoftMax:
			m_func = Softmax;
			break;
		default:
			break;
		}
	}

	const VectorN& GetInput() const
	{
		VectorInOut* vec_in = dynamic_cast<VectorInOut*>(m_input);
		return *(vec_in->m_value);
	}

	VectorN& GetOutput() const
	{
		VectorInOut* vec_out = dynamic_cast<VectorInOut*>(m_output);
		return *(vec_out->m_value);
	}

	virtual void Connect(LayerBase *next)
	{
		LayerBase::Connect(next);

		m_weight = new MatrixMN(this->Size(), this->m_prev->Size());

		m_delta = new VectorN(this->Size());
		m_dw = new MatrixMN(this->Size(), this->m_prev->Size());

		m_sum_delta = new VectorN(this->Size());
		m_sum_dw = new MatrixMN(this->Size(), this->m_prev->Size());

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
		m_middle->Copy(*m_weight * GetInput() + *m_bias);
		m_func(*m_middle, GetOutput());
		if (m_next != nullptr)
		{
			m_next->m_input = m_output;
		}

		//VectorN &outp = GetOutput();
		//for (int i = 0; i < outp.GetSize(); ++i)
		//{
		//	Float c = outp[i];

		//	if (std::abs(c) > 10.0f)
		//	{
		//		std::cout << "c:" << c << endl;
		//	}

		//	if (std::isinf(c) || std::isnan(c))
		//	{
		//		std::cout << "c:" << c << endl;
		//	}
		//}

	}

	virtual void BackProp()
	{
		FullyConnectedLayer *fc = dynamic_cast<FullyConnectedLayer*>(m_next);
		if (fc != nullptr) 
		{
			m_prime_func(*m_middle, *m_middle_prime);
			m_delta->Copy((fc->m_weight->Transpose() * (*fc->m_delta)) ^ (*m_middle_prime));
			m_dw->Copy(*m_delta * GetInput());
		}

		//VectorN &outp = *m_delta;
		//for (int i = 0; i < outp.GetSize(); ++i)
		//{
		//	Float c = outp[i];

		//	if (std::abs(c) > 10.0f)
		//	{
		//		std::cout << "c:" << c << endl;
		//	}

		//	if (std::isinf(c) || std::isnan(c))
		//	{
		//		std::cout << "c:" << c << endl;
		//	}
		//}

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

	virtual void UpdateWeightBias(float eff)
	{	
		*m_weight -= *m_sum_dw * eff;
		*m_bias -= *m_sum_delta * eff;
	}

};
}
#endif //__FULLYCONNECTED_LAYER_H__
