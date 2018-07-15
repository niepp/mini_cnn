#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

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
#include "math/matrix3d.h"
#include "math/mathdef.h"
#include "layer.h"

class ConvolutionalLayer : public LayerBase
{
public:
	//MatrixMN *m_weight;  // m_weight[i][j] : 前一层的第j个神经元到当前层的第i个神经元的连接权重
	//VectorN *m_bias;     // m_bias[i] : 当前层的第i个神经元的偏置
	//VectorN *m_middle;	// middle value


	//VectorN *m_delta;	// equal to dJ/d(bias)
	//MatrixMN *m_dw;		// equal to dJ/d(w)

	//VectorN *m_sum_delta;
	//MatrixMN *m_sum_dw;

	Matrix3D *m_input_img;
	Matrix3D *m_filter;
	size_t m_filter_count;
	VectorN m_bias;

protected:
	eActiveFunc m_activeFuncType;
	ActiveFunc m_activeFunc;
	ActiveFunc m_activePrimeFunc;

public:
	ConvolutionalLayer(uint32_t filterWidth, uint32_t filterHeight, uint32_t filterChannels, eActiveFunc act)
		: LayerBase(filterWidth * filterHeight * filterChannels, new MatrixInOut(), new MatrixInOut())
	{
		//m_bias = new VectorN(neuralCount);
		//m_middle = new VectorN(neuralCount);
		//m_output = new VectorN(neuralCount);

		//m_activeFuncType = act;
		//switch (act)
		//{
		//case eActiveFunc::eSigmod:
		//	m_activeFunc = Sigmoid;
		//	m_activePrimeFunc = SigmoidPrime;
		//	break;
		//case eActiveFunc::eRelu:
		//	m_activeFunc = Relu;
		//	m_activePrimeFunc = ReluPrime;
		//	break;
		//case eActiveFunc::eSoftMax:
		//	m_activeFunc = Softmax;
		//	break;
		//default:
		//	break;
		//}
	}

	virtual void Connect(LayerBase *prev)
	{

		//m_weight = new MatrixMN(this->Size(), prev->Size());

		//m_delta = new VectorN(this->Size());
		//m_dw = new MatrixMN(this->Size(), prev->Size());

		//m_sum_delta = new VectorN(this->Size());
		//m_sum_dw = new MatrixMN(this->Size(), prev->Size());

		//this->m_input = prev->m_output;

	}

	virtual void Init(NormalRandom nrand)
	{
		//for (unsigned int i = 0; i < m_weight->GetRowCount(); ++i)
		//{
		//	for (unsigned int j = 0; j < m_weight->GetColCount(); ++j)
		//	{
		//		(*m_weight)(i, j) = nrand.GetRandom();
		//	}
		//}

		//for (unsigned int i = 0; i < m_bias->GetSize(); ++i)
		//{
		//	(*m_bias)[i] = nrand.GetRandom();
		//}	
	}

	virtual void Forward()
	{

	}

	virtual void BackProp(LayerBase *next)
	{

	}

	virtual void PreTrain()
	{
		//m_sum_delta->MakeZero();
		//m_sum_dw->MakeZero();
	}

	virtual void SumGradient()
	{
		//m_sum_delta->Copy(*m_sum_delta + *m_delta);
		//m_sum_dw->Copy(*m_sum_dw + *m_dw);
	}

	virtual void UpdateWeightBias(float eff)
	{	
		//*m_weight -= *m_sum_dw * eff;
		//*m_bias -= *m_sum_delta * eff;
	}
};

#endif //__CONVOLUTIONAL_LAYER_H__
