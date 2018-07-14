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
#include "fullyconnected_layer.h"
#include "output_layer.h"

class Network
{
	InputLayer* m_inputLayer;
	OutputLayer* m_outputLayer;
	std::vector<LayerBase*> m_layers;
public:
	Network(uint32_t inputCount)
	{
		m_inputLayer = new InputLayer(inputCount);
		m_layers.push_back(m_inputLayer);
	}

	void AddLayer(LayerBase *layer)
	{
		if (m_outputLayer != NULL)
		{
			throw std::exception("add layer after output!");
		}
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

	void Init(NormalRandom nrand)
	{
		for (unsigned int i = 0; i < m_layers.size(); ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->Init(nrand);
		}
	}

	void Forward()
	{
		for (unsigned int i = 1; i < m_layers.size(); ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->Forward();
		}
	}

	void BackProp()
	{
		for (unsigned int i = m_layers.size() - 1; i > 0; --i)
		{
			LayerBase *layer = m_layers[i];
			layer->BackProp();
		}
	}

	void SumGradient()
	{
		for (unsigned int i = m_layers.size() - 1; i > 0; --i)
		{
			LayerBase *layer = m_layers[i];
			layer->SumGradient();
		}
	}

	void UpdateWeightBias(float eff)
	{
		for (unsigned int i = m_layers.size() - 1; i > 0; --i)
		{
			LayerBase *layer = m_layers[i];
			*(layer->m_weight) -= *(layer->m_sum_dw) * eff;
			*(layer->m_bias) -= *(layer->m_sum_delta) * eff;
		}
	}

	void SGD(const std::vector<VectorN*> &batch_img_vec, const std::vector<VectorN*> &batch_label_vec, float eta)
	{
		for (unsigned int i = 1; i < m_layers.size(); ++i)
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


	uint32_t Test(const std::vector<VectorN> &test_img_vec, const std::vector<int> &test_lab_vec)
	{

		assert(test_img_vec.size() == test_lab_vec.size());

		int test_count = test_img_vec.size();
		uint32_t correct = 0;
		for (int k = 0; k < test_count; ++k)
		{
			m_inputLayer->SetInputData(test_img_vec[k]);
			Forward();
			int lab = m_outputLayer->m_output->ArgMax();
			int std_lab = test_lab_vec[k];
			if (lab == std_lab)
			{
				++correct;
			}
		}

		return correct;

	}

};

#endif //__NETWORK_H__

