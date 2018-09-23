#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>

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
		//1. Gaussian initialize
		//	Weights are randomly drawn from Gaussian distributions with fixed mean(e.g., 0) and fixed standard deviation(e.g., 0.01).
		//	This is the most common initialization method in deep learning.
		//2. Xavier initialize

		//3. He initialize
		//  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Delving Deep into Rectifiers : Surpassing Human - Level Performance on ImageNet Classification, Technical report, arXiv, Feb. 2015

		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->Init(nrand);
		}
	}

	void SetTaskCount(int task_count)
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->SetTaskCount(task_count);
		}
	}

	void PreTrain()
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->PreTrain();
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

	void Forward(int task_idx)
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = 0; i < len; ++i)
		{
			LayerBase *layer = m_layers[i];
			layer->Forward(task_idx);
		}
	}

	void BackProp(int task_idx)
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = len - 1; i >= 0; --i)
		{
			LayerBase *layer = m_layers[i];
			layer->BackProp(task_idx);
		}
	}

	void UpdateWeightBias(Float eff)
	{
		int len = static_cast<int>(m_layers.size());
		for (int i = len - 1; i >= 0; --i)
		{
			LayerBase *layer = m_layers[i];
			layer->UpdateWeightBias(eff);
		}
	}

	//void Adagrad(Float eff, Float rho)
	//{
	//	// w' = w' + dw * dw;
	//	// w = w - learning_rate * dw * rho / (rho + sqrt(dw')); 
	//	// forexample: lr»°0.01£¨rho»°3°£

	//	int len = static_cast<int>(m_layers.size());
	//	for (int i = len - 1; i >= 0; --i)
	//	{
	//		LayerBase *layer = m_layers[i];
	//		layer->Adagrad(eff, rho);
	//	}

	//}

	void TrainTask(const std::vector<VectorN*> &batch_img_vec, const std::vector<VectorN*> &batch_label_vec
		, int begin, int end, int task_idx)
	{
		for (int i = begin; i < end; ++i)
		{
			m_inputLayer->SetInputData(*batch_img_vec[i], task_idx);
			m_outputLayer->SetLabelValue(*batch_label_vec[i], task_idx);
			Forward(task_idx);
			BackProp(task_idx);
		}
	}

	void SGD(const std::vector<VectorN*> &batch_img_vec, const std::vector<VectorN*> &batch_label_vec, float eta, const int max_threads)
	{
		assert(batch_img_vec.size() == batch_label_vec.size());
		int batch_size = batch_img_vec.size();		
		int nthreads = std::min(max_threads, batch_size);
		int nstep = (batch_size + nthreads - 1) / nthreads;

		PreTrain();

		std::vector<std::future<void>> futures;
		for (int k = 0; k < nthreads && k * nstep < batch_size; ++k)
		{
			int begin = k * nstep;
			int end = std::min(batch_size, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				TrainTask(batch_img_vec, batch_label_vec, begin, end, k);
			})));
		}
		for (auto &future : futures)
		{
			future.wait();
		}
		float eff = eta / batch_size;
		UpdateWeightBias(eff);

	}

	int TestTask(const std::vector<VectorN*> &test_img_vec, const std::vector<int> &test_lab_vec
		, int begin, int end, int task_idx)
	{
		int c_count = 0;
		for (int i = begin; i < end; ++i)
		{
			m_inputLayer->SetInputData(*test_img_vec[i], task_idx);
			Forward(task_idx);
			int lab = m_outputLayer->GetOutput(task_idx).ArgMax();
			if (lab == test_lab_vec[i])
			{
				++c_count;
			}
		}
		return c_count;
	}

	uInt Test(const std::vector<VectorN*> &test_img_vec, const std::vector<int> &test_lab_vec, const int max_threads)
	{
		assert(test_img_vec.size() == test_lab_vec.size());
		int test_count = test_img_vec.size();

		int nthreads = max_threads;
		int nstep = (test_count + nthreads - 1) / nthreads;

		std::vector<std::future<int>> futures;
		for (int k = 0; k < nthreads && k * nstep < test_count; ++k)
		{
			int begin = k * nstep;
			int end = std::min(test_count, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				return TestTask(test_img_vec, test_lab_vec, begin, end, k);
			})));
		}
		uInt correct = 0;
		for (auto &future : futures)
		{
			correct += future.get();
		}
		return correct;
	}

	Float CalcCost(const std::vector<VectorN*> &img_vec, const std::vector<VectorN*> &lab_vec, const int max_threads)
	{
		assert(img_vec.size() == lab_vec.size());
		int tot_count = img_vec.size();

		int nthreads = max_threads;
		int nstep = (tot_count + nthreads - 1) / nthreads;

		std::vector<std::future<Float>> futures;
		for (int k = 0; k < nthreads && k * nstep < tot_count; ++k)
		{
			int begin = k * nstep;
			int end = std::min(tot_count, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				Float cost = 0;
				for (int i = begin; i < end; ++i)
				{
					m_inputLayer->SetInputData(*img_vec[i], k);
					Forward(k);
					m_outputLayer->SetLabelValue(*lab_vec[i], k);
					Float c = m_outputLayer->GetCost(false, k);
					cost += c;
				}
				return cost;
			})));
		}
		Float tot_cost = 0;
		for (auto &future : futures)
		{
			tot_cost += future.get();
		}
		if (tot_count > 0)
		{
			tot_cost /= tot_count;
		}
		return tot_cost;
	}

	bool GradientCheck(const VectorN &test_img, const VectorN &test_lab, Float absPrecision = 1e-4, Float relatePrecision = 1e-2)
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
				MatrixMN &w = *(fully_layer->m_weight);
				MatrixMN &dw = *(fully_layer->m_dw);
				for (int x = 0; x < w.GetRowCount(); ++x)
				{
					for (int y = 0; y < w.GetColCount(); ++y)
					{
						if (!CalcGradient(test_img, test_lab, w(x, y), dw(x, y), absPrecision, relatePrecision))
						{
							check_ok = false;
						}
					}
				}

				VectorN &b = *(fully_layer->m_bias);
				VectorN &db = *(fully_layer->m_delta);
				for (int x = 0; x < b.GetSize(); ++x)
				{
					if (!CalcGradient(test_img, test_lab, b[x], db[x], absPrecision, relatePrecision))
					{
						check_ok = false;
					}
				}
				//cout << "fully_layer " << std::boolalpha << check_ok << endl;
			}
			else if (conv_layer != nullptr)
			{
				VectorN &b = *(conv_layer->m_bias);
				VectorN &db = *(conv_layer->m_db);
				Int filter_count = conv_layer->m_filters.size();
				for (int k = 0; k < filter_count; ++k)
				{
					Matrix3D &w = *(conv_layer->m_filters[k]);
					Matrix3D &dw = *(conv_layer->m_dw[k]);
					for (int x = 0; x < w.Width(); ++x)
					{
						for (int y = 0; y < w.Height(); ++y)
						{
							for (int c = 0; c < w.Depth(); ++c)
							{
								if (!CalcGradient(test_img, test_lab, w(x, y, c), dw(x, y, c), absPrecision, relatePrecision))
								{
									check_ok = false;
								}
							}
						}
					}

					for (int x = 0; x < b.GetSize(); ++x)
					{
						if (!CalcGradient(test_img, test_lab, b[x], db[x], absPrecision, relatePrecision))
						{
							check_ok = false;
						}
					}
				}

				//cout << "conv_layer " << std::boolalpha << check_ok << endl;

			}
		}
		return check_ok;
	}

	Float CalcLoss(const VectorN &test_img, const VectorN &test_lab)
	{
		m_inputLayer->SetInputData(test_img);
		Forward();
		m_outputLayer->SetLabelValue(test_lab);
		Float loss = m_outputLayer->GetCost(false);
		return loss;
	}

	bool CalcGradient(const VectorN &test_img, const VectorN &test_lab, Float &w, Float &dw, Float absPrecision, Float relatePrecision)
	{
		static const Float EPSILON = 1e-6;

		m_inputLayer->SetInputData(test_img);
		m_outputLayer->SetLabelValue(test_lab);

		Float prev_w = w;
		w = prev_w + EPSILON;
		Forward();
		Float loss_0 = m_outputLayer->GetCost(true);

		w = prev_w - EPSILON;
		Forward();
		Float loss_1 = m_outputLayer->GetCost(true);
		Float delta_by_numerical = (loss_0 - loss_1) / (Float(2) * EPSILON);

		w = prev_w;
		Forward();
		BackProp();

		Float delta_by_bprop = dw;

		if (!f_is_valid(loss_0) || !f_is_valid(loss_1) || !f_is_valid(dw))
		{
			cout << "[overflow] loss_0:" << loss_0 << "\tloss_1:" << loss_1 << "\tdw:" << dw << endl;
			return false;
		}

		Float absError = std::abs(delta_by_bprop - delta_by_numerical);
		bool correct = absError <= absPrecision;
		if (!correct)
		{
			Float relateError = absError / std::abs(delta_by_numerical);
			if (f_is_valid(relateError) && relateError < relatePrecision)
			{
				correct = true;
			}
			else
			{
				cout << "bprop:" << delta_by_bprop << "\tnumerical:" << delta_by_numerical << endl;
			}
		}

		return correct;

	}

};

}

#endif //__NETWORK_H__

