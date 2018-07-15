#ifndef __RESHAPE_LAYER_H__
#define __RESHAPE_LAYER_H__

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

class ReshapeLayer : public LayerBase
{
public:

protected:

public:
	ReshapeLayer(uint32_t neuralCount)
		: LayerBase(neuralCount, new VectorInOut(), new VectorInOut())
	{
		
	}


};

#endif //__RESHAPE_LAYER_H__
