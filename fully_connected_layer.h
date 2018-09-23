#ifndef __FULLYCONNECTED_LAYER_H__
#define __FULLYCONNECTED_LAYER_H__

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

namespace mini_cnn
{
class FullyConnectedLayer : public LayerBase
{
public:
	MatrixMN *m_weight;  // m_weight[i][j] : 前一层的第j个神经元到当前层的第i个神经元的连接权重
	VectorN *m_bias;     // m_bias[i] : 当前层的第i个神经元的偏置
	VectorN *m_middle;	// middle value

	VectorN *m_delta;	// equal to dJ/d(bias)
	MatrixMN *m_dw;		// equal to dJ/d(w)

	struct TaskStorage
	{
		VectorN *m_middle;
		VectorN *m_delta;
		MatrixMN *m_dw;
	};

	std::vector<TaskStorage> m_task_storage;

protected:
	eActiveFunc m_func_type;
	ActiveFunc m_func;
	ActiveFuncDerived m_derived_func;

public:
	FullyConnectedLayer(uInt neuralCount, eActiveFunc act)
		: LayerBase(neuralCount, new VectorInOut(), new VectorInOut())
	{
		m_bias = new VectorN(neuralCount);
		m_middle = new VectorN(neuralCount);

		VectorInOut* vec_out = dynamic_cast<VectorInOut*>(m_output);
		vec_out->m_value = new VectorN(neuralCount);

		m_func_type = act;
		switch (act)
		{
		case eActiveFunc::eSigmod:
			m_func = Sigmoid;
			m_derived_func = SigmoidPrime;
			break;
		case eActiveFunc::eTanh:
			m_func = Tanh;
			m_derived_func = TanhPrime;
			break;
		case eActiveFunc::eRelu:
			m_func = Relu;
			m_derived_func = ReluPrime;
			break;
		case eActiveFunc::eSoftMax:
			m_func = Softmax;
			break;
		default:
			break;
		}
	}

	const VectorN& GetInput() const
	{
		VectorInOut* vec_in = dynamic_cast<VectorInOut*>(m_input);
		return *(vec_in->m_value);
	}

	VectorN& GetOutput() const
	{
		VectorInOut* vec_out = dynamic_cast<VectorInOut*>(m_output);
		return *(vec_out->m_value);
	}

	const VectorN& GetInput(TaskStorageBase &ts) const
	{
		VectorInOut* vec_in = dynamic_cast<VectorInOut*>(ts.m_input);
		return *(vec_in->m_value);
	}

	VectorN& GetOutput(TaskStorageBase &ts) const
	{
		VectorInOut* vec_out = dynamic_cast<VectorInOut*>(ts.m_output);
		return *(vec_out->m_value);
	}

	TaskStorage& GetTaskStorage(int task_idx)
	{
		return m_task_storage[task_idx];
	}

	virtual void Connect(LayerBase *next)
	{
		LayerBase::Connect(next);

		m_weight = new MatrixMN(this->Size(), this->m_prev->Size());

		m_delta = new VectorN(this->Size());
		m_dw = new MatrixMN(this->Size(), this->m_prev->Size());

	}

	virtual void Init(NormalRandom nrand)
	{
		for (unsigned int i = 0; i < m_weight->GetRowCount(); ++i)
		{
			for (unsigned int j = 0; j < m_weight->GetColCount(); ++j)
			{
				(*m_weight)(i, j) = nrand.GetRandom();
			}
		}

		for (unsigned int i = 0; i < m_bias->GetSize(); ++i)
		{
			(*m_bias)[i] = nrand.GetRandom();
		}
	}

	virtual void SetTaskCount(int task_count)
	{
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_delta = new VectorN(this->Size());
			ts.m_dw = new MatrixMN(this->Size(), this->m_prev->Size());
			ts.m_middle = new VectorN(this->Size());
		}
		LayerBase::SetTaskCount(task_count);
	}

	virtual void Forward()
	{
		m_middle->Copy(*m_weight * GetInput() + *m_bias);
		m_func(*m_middle, GetOutput());
		if (m_next != nullptr)
		{
			m_next->m_input = m_output;
		}
	}

	virtual void Forward(int task_idx)
	{
		TaskStorage &ts = m_task_storage[task_idx];
		TaskStorageBase &tsb = GetTaskStorageBase(task_idx);
		VectorN &_m = *ts.m_middle;
		_m.Copy(*m_weight * GetInput(tsb) + *m_bias);
		m_func(_m, GetOutput(tsb));
		if (m_next != nullptr)
		{
			TaskStorageBase &next_tsb = m_next->GetTaskStorageBase(task_idx);
			next_tsb.m_input = tsb.m_output;
		}
	}

	virtual void BackProp(int task_idx)
	{
		TaskStorage &ts = m_task_storage[task_idx];
		TaskStorageBase &tsb = GetTaskStorageBase(task_idx);
		FullyConnectedLayer *fc = dynamic_cast<FullyConnectedLayer*>(m_next);
		if (fc != nullptr)
		{
			VectorN &_d = *ts.m_delta;
			MatrixMN &_w = *ts.m_dw;
			VectorN &_m = *ts.m_middle;
			TaskStorage &next_ts = fc->GetTaskStorage(task_idx);
			m_derived_func(_m);
			_d += (fc->m_weight->Transpose() * (*next_ts.m_delta)) ^ _m;
			_w += _d * GetInput(tsb);
		}
	}

	virtual void PreTrain()
	{
		for (auto& ts : m_task_storage)
		{
			ts.m_delta->MakeZero();
			ts.m_dw->MakeZero();
		}
	}

	virtual void UpdateWeightBias(Float eff)
	{
		m_delta->MakeZero();
		m_dw->MakeZero();
		for (auto& ts : m_task_storage)
		{
			*m_delta += *ts.m_delta;
			*m_dw += *ts.m_dw;
		}
		*m_weight -= *m_dw * eff;
		*m_bias -= *m_delta * eff;
	}

};
}
#endif //__FULLYCONNECTED_LAYER_H__

