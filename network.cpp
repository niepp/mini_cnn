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

class Network
{
	InputLayer* m_inputLayer;
	OutputLayer* m_outputLayer;
	std::vector<LayerBase*> m_layers;
public:
	Network(Uint32 inputCount)
	{
		m_inputLayer = new InputLayer(inputCount);
		m_layers.push_back(m_inputLayer);
	}

	void AddLayer(LayerBase *layer, bool isOutLayer = false)
	{
		if (m_outputLayer != NULL)
		{
			throw std::exception();
		}
		LayerBase* lastLayer = *m_layers.rbegin();
		if (isOutLayer)
		{
			m_outputLayer = dynamic_cast<OutputLayer*>(layer);
		}
		lastLayer->m_prev->Connect(layer);
		m_layers.push_back(layer);
	}

	void SGD(LabelData *training_data, Uint32 n, Uint32 batch_size, float eta, vector<LabelData> &test_data)
	{
		for (Uint32 i = 0; i < m_layers.size(); ++i)
		{
			m_layers[i]->InitWeight();
		}

		random_shuffle(training_data, training_data + n);
		Uint32 batch = n / batch_size;
		for (Uint32 i = 0; i < batch; ++i)
		{
			Uint32 bsize = i == batch - 1 ? n - i * batch_size : batch_size;
			UpdateBatch(training_data + i * batch_size, bsize, eta);
		}

	}

	void UpdateBatch(LabelData *training_data, Uint32 n, Float eta)
	{
		for (Uint32 i = 0; i < n; ++i)
		{
			m_inputLayer->InitInput(training_data[i]);
			m_outputLayer->InitOutput(training_data[i]);
			FeedForward();
			Backprop(training_data[i]);
			UpdateWeight(eta);
		}
	}

	void Backprop(LabelData &training_data)
	{
		for (Uint32 i = m_layers.size() - 1; i > 0 ; --i)
		{
			m_layers[i]->BackProp();
		}
	}

	void FeedForward()
	{
		for (Uint32 i = 0; i < m_layers.size(); ++i)
		{
			m_layers[i]->Forward();
		}
	}

	void UpdateWeight(float eta)
	{
		for (Uint32 i = m_layers.size() - 1; i > 0 ; --i)
		{
			m_layers[i]->UpdateWeight(eta);
		}
	}

};

