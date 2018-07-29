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

class FilterDimension
{
public:
	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_channels;
	uint32_t m_padding;
	uint32_t m_stride_w;
	uint32_t m_stride_h;
	FilterDimension(uint32_t w, uint32_t h, uint32_t c, uint32_t padding, uint32_t stride_w, uint32_t stride_h) :
		m_width(w), m_height(h), m_channels(c), m_padding(padding), m_stride_w(stride_w), m_stride_h(stride_h)
	{
	}
};


/*
for the l-th Conv layer:
	(l)     (l-1)      (l)    (l)
   Z	=  X       *  W    + B
    (l)      (l)
   X    = f(Z   )
*/
class ConvolutionalLayer : public LayerBase
{
public:
	std::vector<Matrix3D*> m_filters;
	VectorN *m_bias;

	VectorN *m_db;	// dJ/d(bias)
	std::vector<Matrix3D*> m_dw;	// dJ/d(w)
	Matrix3D *m_delta;		// dJ/d(z)

protected:
	Matrix3D *m_middle;	// middle value

	Matrix3D *m_input_img;
	Matrix3D *m_output_img;
	Matrix3D *m_middle_prime;

	// mini-batch中每个batch里的多个训练样本的梯度之和
	VectorN *m_sum_db;
	std::vector<Matrix3D*> m_sum_dw;

protected:
	FilterDimension m_filterDim;
	eActiveFunc m_activeFuncType;
	MatActiveFunc m_activeFunc;
	MatActiveFunc m_activePrimeFunc;

public:
	ConvolutionalLayer(uint32_t filterCount, uint32_t filterWidth, uint32_t filterHeight, uint32_t filterChannels, uint32_t padding, uint32_t stride_w, uint32_t stride_h, eActiveFunc act)
		: LayerBase(0, new MatrixInOut(), new MatrixInOut())
		, m_filterDim(filterWidth, filterHeight, filterChannels, padding, stride_w, stride_h)
	{

		for (uint32_t i = 0; i < filterCount; ++i)
		{
			m_filters.push_back(new Matrix3D(filterWidth, filterHeight, filterChannels));
		}
		m_bias = new VectorN(filterCount);

		for (uint32_t i = 0; i < filterCount; ++i)
		{
			m_dw.push_back(new Matrix3D(filterWidth, filterHeight, filterChannels));
		}
		m_db = new VectorN(filterCount);

		for (uint32_t i = 0; i < filterCount; ++i)
		{
			m_sum_dw.push_back(new Matrix3D(filterWidth, filterHeight, filterChannels));
		}
		m_sum_db = new VectorN(filterCount);

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
		default:
			break;
		}
	}

	const Matrix3D& GetInput() const
	{
		MatrixInOut* mat_in = dynamic_cast<MatrixInOut*>(m_input);
		return *(mat_in->m_value);
	}

	Matrix3D& GetOutput() const
	{
		MatrixInOut* mat_out = dynamic_cast<MatrixInOut*>(m_output);
		return *(mat_out->m_value);
	}

	virtual void Connect(LayerBase *next)
	{
		LayerBase::Connect(next);

		MatrixInOut* pre_out = dynamic_cast<MatrixInOut*>(m_prev->m_output);
		auto input_img = pre_out->m_value;

		// calc output size
		// Padding::Valid
		uint32_t input_w = input_img->Width();
		uint32_t input_h = input_img->Height();
		uint32_t nw = static_cast<uint32_t>(floorf(1.0f * (input_w - m_filterDim.m_width) / m_filterDim.m_stride_w)) + 1;
		uint32_t nh = static_cast<uint32_t>(floorf(1.0f * (input_h - m_filterDim.m_height) / m_filterDim.m_stride_h)) + 1;
		uint32_t nd = m_filters.size();

		m_neuralCount = nw * nh * nd;

		MatrixInOut* this_out = dynamic_cast<MatrixInOut*>(m_output);
		this_out->m_value = new Matrix3D(nw, nh, nd);
		m_output_img = this_out->m_value;

		m_delta = new Matrix3D(nw, nh, nd);

		m_middle = new Matrix3D(nw, nh, nd);
		m_middle_prime = new Matrix3D(nw, nh, nd);

	}

	virtual void Init(NormalRandom nrand)
	{
		for (uint32_t k = 0; k < m_filters.size(); ++k)
		{
			auto filter = m_filters[k];
			uint32_t w = filter->Width();
			uint32_t h = filter->Height();
			uint32_t d = filter->Depth();
			for (uint32_t i = 0; i < w; ++i)
			{
				for (uint32_t j = 0; j < h; ++j)
				{
					for (uint32_t c = 0; c < d; ++c)
					{
						filter->operator()(i, j, c) = nrand.GetRandom();
					}
				}
			}
		}

		for (uint32_t i = 0; i < m_bias->GetSize(); ++i)
		{
			(*m_bias)[i] = nrand.GetRandom();
		}

	}

	virtual void Forward()
	{
		MatrixInOut* pre_out = dynamic_cast<MatrixInOut*>(m_prev->m_output);
		m_input_img = pre_out->m_value;

		m_input_img->Conv(m_middle, m_filters, m_filterDim.m_stride_w, m_filterDim.m_stride_h, 0, 0, Padding::Valid);
		m_middle->AddBias(*m_bias);
		m_activeFunc(*m_middle, *m_output_img);

		assert(m_next != nullptr);
		if (m_next->m_input->m_type != m_output->m_type)
		{
			VectorInOut* next_vec_in = dynamic_cast<VectorInOut*>(m_next->m_input);
			next_vec_in->m_value = m_output_img->Flatten();
		}

	}

	virtual void BackProp()
	{
		FullyConnectedLayer *fully_layer = dynamic_cast<FullyConnectedLayer*>(m_next);
		ConvolutionalLayer *conv_layer = dynamic_cast<ConvolutionalLayer*>(m_next);

		m_activePrimeFunc(*m_middle, *m_middle_prime);

		if (fully_layer != nullptr)
		{
			auto flatten_delta = (fully_layer->m_weight->Transpose() * (*fully_layer->m_delta)) ^ (*(m_middle_prime->Flatten()));
			m_delta->Copy(*flatten_delta.Unflatten(m_delta->Width(), m_delta->Height(), m_delta->Depth()));			
		}
		else if (conv_layer != nullptr)
		{
			conv_layer->m_delta->ConvDepthWise(m_delta, conv_layer->m_filters,
				conv_layer->m_filterDim.m_stride_w, conv_layer->m_filterDim.m_stride_h, 
				conv_layer->m_filterDim.m_width - 1, conv_layer->m_filterDim.m_height - 1,
				Padding::Valid);

			(*m_delta)^= (*m_middle_prime);

		}
		else
		{
			throw new exception("no implement!");
		}

		m_input_img->ConvDepthWise(m_dw, *m_delta, m_filterDim.m_stride_w, m_filterDim.m_stride_h, 0, 0, Padding::Valid);

		assert(m_filters.size() == m_bias->GetSize() &&
			m_bias->GetSize() == m_delta->Depth());

		for (int i = 0; i < m_filters.size(); ++i)
		{
			(*m_bias)[i] = m_delta->SumByDepthWise(i);
		}
	}

	virtual void PreTrain()
	{
		m_sum_db->MakeZero();
		for (uint32_t i = 0; i < m_filters.size(); ++i)
		{
			m_sum_dw[i]->MakeZero();
		}
	}

	virtual void SumGradient()
	{
		m_sum_db->Copy(*m_sum_db + *m_db);
		for (uint32_t i = 0; i < m_filters.size(); ++i)
		{
			*m_sum_dw[i] += *m_dw[i];
		}
	}

	virtual void UpdateWeightBias(float eff)
	{
		*m_bias -= *m_sum_db * eff;
		for (uint32_t i = 0; i < m_filters.size(); ++i)
		{
			*m_sum_dw[i] *= eff;
			*m_filters[i] -= *m_sum_dw[i];
		}
	}

};

#endif //__CONVOLUTIONAL_LAYER_H__

