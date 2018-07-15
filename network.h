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
		m_outputLayer->BackProp();
		for (int i = (int)m_layers.size() - 2; i >= 0; --i)
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


	uint32_t Test(const std::vector<VectorN> &test_img_vec, const std::vector<int> &test_lab_vec)
	{

		assert(test_img_vec.size() == test_lab_vec.size());

		int test_count = test_img_vec.size();
		uint32_t correct = 0;
		for (int k = 0; k < test_count; ++k)
		{
			m_inputLayer->SetInputData(test_img_vec[k]);
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

};

#endif //__NETWORK_H__

