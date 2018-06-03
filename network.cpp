#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>  
using namespace std;

#include "mathconsts.h"
#include "vectorn.h"
#include "matrixmxn.h"
#include "types.h"

float Random()
{
	return (Float)rand() / RAND_MAX;
}

float Sigmoid(float z)
{
	return 1.0f / (1.0f + exp(-z));
}

float SigmoidPrime(float z)
{
	return Sigmoid(z) * (1.0f - Sigmoid(z));
}

struct LabelData
{
	Float data[784];
	Float a;
};


class LayerBase
{
protected:
	MatrixMN m_weight;  // m_weight[i][j] : 前一层的第j个神经元到当前层的第i个神经元的连接权重
	VectorN m_bias;     // m_bias[i] : 当前层的第i个神经元的偏置
	VectorN m_delta;
	VectorN m_weightedInput;
	VectorN m_output;    // m_output = Sigmoid(m_weightedInput)

	Uint32 m_neuralCount;	
public: 
	LayerBase *m_prev, *m_next;

	LayerBase(Uint32 neuralCount) : m_neuralCount(neuralCount), m_prev(NULL), m_next(NULL)
	{
		m_delta.SetSize(neuralCount);
		m_weightedInput.SetSize(neuralCount);
		m_output.SetSize(neuralCount);
	}

	Uint32 Size()
	{
		return m_neuralCount;
	}

	MatrixMN& Weight()
	{
		return m_weight;
	}

	VectorN& Bias()
	{
		return m_bias;
	}

	VectorN& Delta()
	{
		return m_delta;
	}

	virtual VectorN& OutPut() = 0;

	virtual void Connect(LayerBase *next)
	{
		next->m_prev = this;
		this->m_next = next;
	}

	virtual void InitWeight()
	{

	}

	virtual void Forward()
	{

	}

	virtual void BackProp()
	{

	}

	virtual void UpdateWeight(float eta)
	{

	}

};

class InputLayer : public LayerBase
{
protected:
	VectorN m_inputData;
public:
	InputLayer(Uint32 neuralCount) : LayerBase(neuralCount), m_inputData(neuralCount, 0.0f)
	{
	}

	void InitInput(LabelData &training_data)
	{
		for (Uint32 i = 0; i < m_neuralCount; ++i)
		{
			m_inputData[i] = training_data.data[i];
			m_output[i] = m_inputData[i];
		}
	}

	virtual VectorN& OutPut()
	{
		return m_output;
	}
};

class FullyConnectedLayer : public LayerBase
{
	
public:
	FullyConnectedLayer(Uint32 neuralCount) : LayerBase(neuralCount)
	{
	}

	virtual void Connect(LayerBase *next)
	{
		next->m_prev = this;
		this->m_next = next;

		m_weight.SetSize(this->Size(), this->m_prev->Size());
		m_bias.SetSize(this->Size());

	}

	virtual VectorN& OutPut()
	{
		return m_output;
	}

	virtual void InitWeight()
	{
		Uint32 row = m_weight.GetRowCount();
		Uint32 col = m_weight.GetColCount();
		for (Uint32 i = 0; i < row; ++i)
		{
			for (Uint32 j = 0; j < col; ++j)
			{
				m_weight(i, j) = Random();
			}
			m_bias[i] = Random();
		}
	}

	virtual void Forward()
	{
		VectorN &prevOut = m_prev->OutPut();
		Uint32 inputCount = prevOut.GetSize();
		for (Uint32 i = 0; i < m_neuralCount; ++i)
		{
			Float sum = 0;
			for (Uint32 j = 0; j < inputCount; ++j)
			{
				sum += m_weight(i, j) * prevOut[j];
			}
			m_weightedInput[i] = sum + m_bias[i];
			m_output[i] = Sigmoid(m_weightedInput[i]);
		}
	}

	virtual void BackProp()
	{
		MatrixMN &w = m_next->Weight();
		VectorN &d = m_next->Delta();
		for (Uint32 i = 0; i < m_neuralCount; ++i)
		{
			Float s = 0;
			Uint32 row = w.GetRowCount();
			for (Uint32 j = 0; j < row; ++j)
			{
				s += w(j, i) * d[j];
			}
			m_delta[i] = s * SigmoidPrime(m_weightedInput[i]);
		}
	}

	virtual void UpdateWeight(float eta)
	{
		VectorN &prevOut = m_prev->OutPut();
		Uint32 inputCount = prevOut.GetSize();
		Uint32 row = m_delta.GetSize();
		for (Uint32 i = 0; i < row; ++i)
		{
			for (Uint32 j = 0; j < inputCount; ++j)
			{
				m_weight(i, j) += eta * prevOut[j] * m_delta[i];
			}
			m_bias[i] += eta * m_delta[i];
		}		
	}

};

class OutputLayer : public FullyConnectedLayer
{
	VectorN m_reference;  // 已知的参考值
public:
	OutputLayer(Uint32 neuralCount) : FullyConnectedLayer(neuralCount)
	{
	}

	void InitOutput(LabelData &training_data)
	{
		for (Uint32 i = 0; i < m_neuralCount; ++i)
		{
			m_reference[i] = training_data.a;  // ???
		}
	}

	virtual void BackProp()
	{
		for (Uint32 i = 0; i < m_neuralCount; ++i)
		{
			m_delta[i] = (Sigmoid(m_weightedInput[i]) - m_reference[i]) * SigmoidPrime(m_weightedInput[i]) ;
		}
	}

};


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

int main()
{
	Uint32 n = 0;
	LabelData *trainData = NULL;
	std::vector<LabelData> testData;

	Network nn(487);
	nn.AddLayer(new FullyConnectedLayer(32));
	nn.AddLayer(new OutputLayer(10));
	nn.SGD(trainData, n, 20, 0.2f, testData);
	return 0;
}

