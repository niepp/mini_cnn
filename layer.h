#ifndef __LAYER_H__
#define __LAYER_H__

namespace mini_cnn
{
enum activation_type
{
	eSigmod,
	eTanh,
	eRelu,
	eSoftMax,
};

enum lossfunc_type
{
	eMSE,
	eSigmod_CrossEntropy,
	eSoftMax_LogLikelihood,
};

// z = w * a + b
// a = f(z)
class layer_base
{
protected:
	layer_base* m_next;
	layer_base* m_prev;
public:
	int_t m_out_size;
	varray m_w;          // weight vector
	varray m_b;          // bias vector

protected:
	struct task_storage
	{
		varray m_dw;
		varray m_db;
		varray m_z;      // z vector
		varray m_a;      // output vector
		varray m_delta;
		varray m_wd;	 // w' X delta
	};

	std::vector<task_storage> m_task_storage;

public:
	layer_base(int_t out_size)
		: m_out_size(out_size)
	{
	}

	const varray& output(int_t task_idx) const
	{
		return m_task_storage[task_idx].m_a;
	}

	task_storage& get_task_storage(int_t task_idx)
	{
		return m_task_storage[task_idx];
	}

	virtual void init_weight(normal_random nrand)
	{
		int_t w_sz = m_w.size();
		for (int_t i = 0; i < w_sz; ++i)
		{
			m_w[i] = nrand.get_random();
		}

		int_t b_sz = m_b.size();
		for (int_t i = 0; i < b_sz; ++i)
		{
			m_b[i] = nrand.get_random();
		}
	}

	virtual void connect(layer_base *next)
	{
		if (next != nullptr)
		{
			next->m_prev = this;
			m_next = next;
		}
	}

	virtual void set_task_count(int task_count) = 0;

	/*
		input: input of this layer
	*/
	virtual const varray& forw_prop(const varray& input, int_t task_idx) = 0;

	/*
		next_wd: next layer's transpose(weight) * delta
	*/
	virtual const varray& back_prop(const varray& next_wd, int_t task_idx) = 0;

	virtual void pre_train()
	{
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.make_zero();
			ts.m_db.make_zero();
		}
	}

	virtual void update_weights(float_t eff)
	{
		int_t b_sz = m_b.size();
		int_t w_sz = m_w.size();

		// merge weights from all task
		int_t task_count = (int_t)m_task_storage.size();
		auto& ts_base = m_task_storage[0];
		for (int_t k = 1; k < task_count; ++k)
		{
			auto& ts = m_task_storage[k];
			for (int_t i = 0; i < b_sz; ++i)
			{
				ts_base.m_db[i] += ts.m_db[i];
			}
			for (int_t i = 0; i < w_sz; ++i)
			{
				ts_base.m_dw[i] += ts.m_dw[i];
			}
		}

		// update weights
		for (int_t i = 0; i < b_sz; ++i)
		{
			m_b[i] -= ts_base.m_db[i] * eff;
		}
		for (int_t i = 0; i < w_sz; ++i)
		{
			m_w[i] -= ts_base.m_dw[i] * eff;
		}

	}

};
}
#endif //__LAYER_H__
