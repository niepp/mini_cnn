#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__

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
#include "layer.h"

class FlattenLayer : public LayerBase
{
public:

protected:

public:
	FlattenLayer(uint32_t neuralCount)
		: LayerBase(neuralCount, new VectorInOut(), new VectorInOut())
	{

	}

};

#endif //__FLATTEN_LAYER_H__
