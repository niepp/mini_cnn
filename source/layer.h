#ifndef __LAYER_H__
#define __LAYER_H__

namespace mini_cnn
{

enum class padding_type
{
	eValid, // only use valid pixels of input
	eSame   // padding zero around input to keep image size
};

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

class shape3d
{
public:
	nn_int m_w;
	nn_int m_h;
	nn_int m_d;
	shape3d() : m_w(0), m_h(0), m_d(0)
	{
	}

	shape3d(nn_int w, nn_int h, nn_int d) : m_w(w), m_h(h), m_d(d)
	{
	}

	void set(nn_int w, nn_int h, nn_int d)
	{
		m_w = w;
		m_h = h;
		m_d = d;
	}

	void reshape(nn_int w, nn_int h, nn_int d)
	{
		nn_assert(m_w * m_h * m_d == w * h * d);
		set(w, h, d);
	}

	nn_int size() const
	{
		nn_assert(m_w * m_h * m_d > 0);
		return m_w * m_h * m_d;
	}

	bool is_img() const
	{
		return m_h > 1 || m_d > 1;
	}
};

// z = w * x + b
// x = f(z)
class layer_base
{
protected:
	layer_base* m_next;
	layer_base* m_prev;
public:
	shape3d m_out_shape;
	varray m_w;          // weight vector
	varray m_b;          // bias vector

protected:
	struct task_storage
	{
		varray m_dw;
		varray m_db;
		varray m_z;      // z vector
		varray m_x;      // output vector
		varray m_delta;
		varray m_wd;	 // w' * delta
	};

	std::vector<task_storage> m_task_storage;

public:
	layer_base()
	{
	}

	nn_int out_size() const
	{
		return m_out_shape.size();
	}

	nn_int paramters_count() const
	{
		return m_w.size() + m_b.size();
	}

	const varray& get_output(nn_int task_idx) const
	{
		return m_task_storage[task_idx].m_x;
	}

	task_storage& get_task_storage(nn_int task_idx)
	{
		return m_task_storage[task_idx];
	}

	void clear_grident()
	{
		for (auto &ts : m_task_storage)
		{
			ts.m_dw.make_zero();
			ts.m_db.make_zero();
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

	virtual nn_int fan_in_size() const
	{
		return out_size();
	}

	virtual nn_int fan_out_size() const
	{
		return out_size();
	}

	virtual void set_task_count(nn_int task_count) = 0;

	/*
		input: input of this layer
	*/
	virtual void forw_prop(const varray &input, nn_int task_idx) = 0;

	/*
		next_wd: next layer's transpose(weight) * delta
	*/
	virtual void back_prop(const varray &next_wd, nn_int task_idx) = 0;

	virtual void update_weights(nn_float eff)
	{
		nn_int b_sz = m_b.size();
		nn_int w_sz = m_w.size();

		// merge weights from all task
		nn_int task_count = (nn_int)m_task_storage.size();
		auto& ts_sum = m_task_storage[0];
		
		nn_float *nn_restrict vec_sum_db = &ts_sum.m_db[0];
		nn_float *nn_restrict vec_sum_dw = &ts_sum.m_dw[0];

		for (nn_int k = 1; k < task_count; ++k)
		{
			auto& ts = m_task_storage[k];
			const nn_float *nn_restrict vec_task_db = &ts.m_db[0];
			for (nn_int i = 0; i < b_sz; ++i)
			{
				vec_sum_db[i] += vec_task_db[i];
			}

			const nn_float *nn_restrict vec_task_dw = &ts.m_dw[0];
			for (nn_int i = 0; i < w_sz; ++i)
			{
				vec_sum_dw[i] += vec_task_dw[i];
			}
		}

		// update weights
		nn_float *nn_restrict vec_b = &m_b[0];
		for (nn_int i = 0; i < b_sz; ++i)
		{
			vec_b[i] -= vec_sum_db[i] * eff;
		}

		nn_float *nn_restrict vec_w = &m_w[0];
		for (nn_int i = 0; i < w_sz; ++i)
		{
			vec_w[i] -= vec_sum_dw[i] * eff;
		}

		// clear task storage
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.make_zero();
			ts.m_db.make_zero();
		}

	}

};
}
#endif //__LAYER_H__
