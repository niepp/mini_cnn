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
	InOut *m_input;
	InOut *m_output;
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

	virtual void Forward()
	{
	}

	virtual void BackProp()
	{
	}

	virtual void PreTrain()
	{
	}

	virtual void SumGradient()
	{
	}

	virtual void UpdateWeightBias(float eff)
	{
	}

};
}
#endif //__LAYER_H__
