#ifndef __INPUT_OUTPUT_H__
#define __INPUT_OUTPUT_H__

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

namespace mini_cnn
{
enum InOutType
{
	IO_Vector,
	IO_Matrix,
};

class InOut
{
public:
	InOutType m_type;
public:
	InOut(InOutType typ) : m_type(typ)
	{
	}
	virtual ~InOut()
	{
	}
	//static InOut* Create(InOutType typ)
	//{
	//	switch (typ)
	//	{
	//	case IO_Vector:
	//		return new VectorInOut();
	//		break;
	//	case IO_Matrix:
	//		return new MatrixInOut();
	//		break;
	//	default:
	//		break;
	//	}
	//}
};

class VectorInOut : public InOut
{
public:	
	VectorN *m_value;
public:
	VectorInOut() : InOut(InOutType::IO_Vector)
	{}
};

class MatrixInOut : public InOut
{
public:
	Matrix3D *m_value;
public:
	MatrixInOut() : InOut(InOutType::IO_Matrix)
	{}
};
}
#endif //__INPUT_OUTPUT_H__
