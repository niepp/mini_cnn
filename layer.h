#ifndef __LAYER_H__
#define __LAYER_H__

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
#include "input_output.h"
namespace mini_cnn
{
enum eActiveFunc
{
	eSigmod,
	eTanh,
	eRelu,
	eSoftMax,
};

enum eLossFunc
{
	eMSE,
	eSigmod_CrossEntropy,
	eSoftMax_LogLikelihood,
};

// z = w * a + b
// a = f(z)
class LayerBase
{
public:
	// input of this layer, this is a ref to prev layer's output
	InOut *m_input;
	InOut *m_output;

	struct TaskStorageBase
	{
		InOut *m_input;
		InOut *m_output;
	};

	std::vector<TaskStorageBase> m_task_storage_base;

protected:
	uInt m_neuralCount;	
	LayerBase *m_prev;
	LayerBase *m_next;

public:
	LayerBase(uInt neuralCount, InOut *input, InOut *output)
		: m_neuralCount(neuralCount), m_input(input), m_output(output)
	{
	}

	uInt Size()
	{
		return m_neuralCount;
	}

	TaskStorageBase& GetTaskStorageBase(int task_idx)
	{
		return m_task_storage_base[task_idx];
	}

	virtual void Connect(LayerBase *next)
	{
		if (next != nullptr)
		{
			next->m_prev = this;
			this->m_next = next;
		}
	}

	virtual void Init(NormalRandom nrand)
	{
	}

	virtual void SetTaskCount(int task_count)
	{
		m_task_storage_base.resize(task_count);
		for (auto& ts : m_task_storage_base)
		{
			VectorInOut *vec_in = new VectorInOut();
			ts.m_input = vec_in;

			VectorInOut *vec_out = new VectorInOut();
			vec_out->m_value = new VectorN(m_neuralCount);
			ts.m_output = vec_out;
		}
	}

	virtual void Forward()
	{
	}

	virtual void BackProp()
	{
	}

	virtual void Forward(int task_idx)
	{
	}

	virtual void BackProp(int task_idx)
	{
	}

	virtual void PreTrain()
	{
	}

	virtual void UpdateWeightBias(Float eff)
	{
	}

};
}
#endif //__LAYER_H__
