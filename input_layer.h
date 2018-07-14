#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__

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

class InputLayer : public LayerBase
{
public:
	InputLayer(uint32_t neuralCount) : LayerBase(neuralCount)
	{
		m_output = new VectorN(neuralCount);
	}
	
	void SetInputData(const VectorN &input)
	{
		m_output->Copy(input);
	}

};

#endif //__INPUT_LAYER_H__
