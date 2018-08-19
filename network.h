#ifndef __NETWORK_H__
#define __NETWORK_H__

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
#include "input_layer.h"
#include "fully_connected_layer.h"
#include "convolutional_layer.h"
#include "output_layer.h"

namespace mini_cnn
{
class Network
{
	InputLayer* m_inputLayer;
	OutputLayer* m_outputLayer;
	std::vector<LayerBase*> m_layers;
public:
	Network()
	{
	}

	void AddLayer(LayerBase *layer)
	{
		if (m_outputLayer != NULL)
		{
			throw std::exception("add layer after output!");
		}

		if (m_inputLayer == nullptr)
		{
			InputLayer *inl = dynamic_cast<InputLayer*>(layer);
			if (inl == nullptr)
			{
				throw std::exception("must add input layer first!");
			}	
			m_layers.push_back(layer);
			m_inputLayer = inl;
		}
		else
		{
			LayerBase* lastLayer = *m_layers.rbegin();
			lastLayer->Connect(layer);
			m_layers.push_back(layer);

			OutputLayer *ol = dynamic_cast<OutputLayer*>(layer);
			if (ol != nullptr)
			{
				m_outputLayer = ol;
				m_outputLayer->Connect(nullptr);
			}
		}
	}

	void Init(NormalRandom nrand)
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->Init(nrand);
		}
	}

	void Forward()
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->Forward();
		}
	}

	void BackProp()
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = len - 1; i >= 0; --i)
		{
			LayerBase *layer = m_layers[i];
			layer->BackProp();
		}
	}

	void SumGradient()
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = len - 1; i >= 0; --i)
		{
			LayerBase *layer = m_layers[i];
			layer->SumGradient();
		}
	}

	void UpdateWeightBias(float eff)
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = len - 1; i >= 0; --i)
		{
			LayerBase *layer = m_layers[i];
			layer->UpdateWeightBias(eff);
		}
	}

	void SGD(const std::vector<VectorN*> &batch_img_vec, const std::vector<VectorN*> &batch_label_vec, float eta)
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->PreTrain();
		}

		assert(batch_img_vec.size() == batch_label_vec.size());

		unsigned int batch_size = batch_img_vec.size();
		for (unsigned int k = 0; k < batch_size; ++k)
		{
			m_inputLayer->SetInputData(*batch_img_vec[k]);
			m_outputLayer->SetLabelValue(*batch_label_vec[k]);
			Forward();
			BackProp();
			SumGradient();
		}

		float eff = eta / batch_size;
		UpdateWeightBias(eff);

	}


	uInt Test(const std::vector<VectorN*> &test_img_vec, const std::vector<int> &test_lab_vec)
	{

		assert(test_img_vec.size() == test_lab_vec.size());

		int test_count = test_img_vec.size();
		uInt correct = 0;
		for (int k = 0; k < test_count; ++k)
		{
			m_inputLayer->SetInputData(*test_img_vec[k]);
			Forward();
			int lab = m_outputLayer->GetOutput().ArgMax();
			int std_lab = test_lab_vec[k];
			if (lab == std_lab)
			{
				++correct;
			}
		}

		return correct;

	}

	Float CalcCost(const std::vector<VectorN*> &img_vec, const std::vector<VectorN*> &lab_vec)
	{
		assert(img_vec.size() == lab_vec.size());
		int tot_count = img_vec.size();
		Float tot_cost = 0;
		for (int k = 0; k < tot_count; ++k)
		{
			m_inputLayer->SetInputData(*img_vec[k]);
			Forward();
			m_outputLayer->SetLabelValue(*lab_vec[k]);
			Float c = m_outputLayer->GetCost();
			tot_cost += c;
		}
		if (tot_count > 0)
		{
			tot_cost /= tot_count;
		}
		return tot_cost;
	}

	bool GradientCheck(const VectorN &test_img, const VectorN &test_lab, Float precision = 1e-4)
	{
		assert(!m_layers.empty());
		bool check_ok = true;
		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			FullyConnectedLayer *fully_layer = dynamic_cast<FullyConnectedLayer*>(layer);
			ConvolutionalLayer *conv_layer = dynamic_cast<ConvolutionalLayer*>(layer);
			if (fully_layer != nullptr)
			{
				MatrixMN &w = *fully_layer->m_weight;
				MatrixMN &dw = *fully_layer->m_dw;
				for (int x = 0; x < w.GetRowCount(); ++x)
				{
					for (int y = 0; y < w.GetColCount(); ++y)
					{
						if (!CalcDelta(test_img, test_lab, precision, w(x, y), dw(x, y)))
						{
							check_ok = false;
						}
					}
				}

				VectorN &b = *fully_layer->m_bias;
				VectorN &db = *fully_layer->m_delta;
				for (int x = 0; x < b.GetSize(); ++x)
				{
					if (!CalcDelta(test_img, test_lab, precision, b[x], db[x]))
					{
						check_ok = false;
					}
				}
			}
		}
		return check_ok;
	}

	Float CalcLoss(const VectorN &test_img, const VectorN &test_lab)
	{
		m_inputLayer->SetInputData(test_img);
		Forward();
		m_outputLayer->SetLabelValue(test_lab);
		Float loss = m_outputLayer->GetCost();
		return loss;
	}

	bool CalcDelta(const VectorN &test_img, const VectorN &test_lab, Float precision, Float &w, Float &dw)
	{
		static const Float EPSILON = 1e-9;

		m_inputLayer->SetInputData(test_img);
		m_outputLayer->SetLabelValue(test_lab);

		Float prev_w = w;
		w = prev_w + EPSILON;
		Forward();
		Float loss_0 = m_outputLayer->GetCost();

		w = prev_w - EPSILON;
		Forward();
		Float loss_1 = m_outputLayer->GetCost();
		Float delta_by_numerical = (loss_0 - loss_1) / (Float(2) * EPSILON);

		w = prev_w;
		Forward();
		BackProp();

		Float delta_by_bprop = dw;

		bool correct = std::abs(delta_by_bprop - delta_by_numerical) <= precision;
		if (!correct)
		{
			int kkk = 0;
		}
		return correct;

	}

};

}

#endif //__NETWORK_H__

