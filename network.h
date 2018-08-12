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


	uint32_t Test(const std::vector<VectorN*> &test_img_vec, const std::vector<int> &test_lab_vec)
	{

		assert(test_img_vec.size() == test_lab_vec.size());

		int test_count = test_img_vec.size();
		uint32_t correct = 0;
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

	float32_t CalcCost(const std::vector<VectorN*> &img_vec, const std::vector<VectorN*> &lab_vec)
	{
		assert(img_vec.size() == lab_vec.size());
		int tot_count = img_vec.size();
		float32_t tot_cost = 0;
		for (int k = 0; k < tot_count; ++k)
		{
			m_inputLayer->SetInputData(*img_vec[k]);
			Forward();
			m_outputLayer->SetLabelValue(*lab_vec[k]);
			float32_t c = m_outputLayer->GetCost();
			tot_cost += c;
		}
		if (tot_count > 0)
		{
			tot_cost /= tot_count;
		}
		return tot_cost;
	}

	bool GradientCheck(const VectorN &test_img, const VectorN &test_lab, float32_t eps = 1e-4)
	{
		assert(!m_layers.empty());
		assert(test_img.GetSize() == test_lab.GetSize());

		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			/*for (int i = 0; i < (int)w.size(); i++) {
				if (calc_delta_diff(in, &v[0], data_size, w, dw, i) > eps) return false;
			}
			for (int i = 0; i < (int)b.size(); i++) {
				if (calc_delta_diff(in, &v[0], data_size, b, db, i) > eps) return false;
			}*/
		}
		
		return true;
	}

};

#endif //__NETWORK_H__

