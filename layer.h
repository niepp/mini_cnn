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
#include "math/mathdef.h"

enum eActiveFunc
{
	eSigmod,
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
	uint32_t m_neuralCount;	
	const VectorN *m_input;		// input of this layer, this is a ref to prev layer's output
	VectorN *m_output;			// output of this layer
protected:
	VectorN *m_outputPrime;

public:
	LayerBase(uint32_t neuralCount) : m_neuralCount(neuralCount)
	{
		m_outputPrime = new VectorN(neuralCount);
	}

	uint32_t Size()
	{
		return m_neuralCount;
	}

	virtual void Connect(LayerBase *prev)
	{
	}

	virtual void Init(NormalRandom nrand)
	{
	}

	virtual void Forward()
	{
	}

	virtual void BackProp(LayerBase *next)
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

#endif //__LAYER_H__
