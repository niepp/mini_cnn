#ifndef __LAYER_H__
#define __LAYER_H__

namespace mini_cnn
{

enum phase_type
{
	eTrain,
	eTest,
	eGradientCheck,
};

enum padding_type
{
	eValid, // only use valid pixels of input
	eSame   // padding zero around input to keep image size
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

	activation_base *m_activation;
	phase_type m_phase_type;

public:
	shape3d m_out_shape;
	varray m_w;          // weight matrix
	varray m_b;          // bias vector

protected:
	varray m_z_vec;      // z of a batch
	varray m_x_vec;      // output of a batch
	varray m_wd_vec;	 // w' * delta of a batch

	struct task_storage
	{
		varray m_dw;
		varray m_db;
		varray m_delta;
	};
	std::vector<task_storage> m_task_storage;
	nn_int m_task_count;

public:
	layer_base(activation_base *activation = nullptr) : m_activation(activation)
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

	const varray& get_output() const
	{
		return m_x_vec;
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

	virtual void set_phase_type(phase_type phase)
	{
		m_phase_type = phase;
	}

	virtual void set_fixed_prop(nn_int task_idx)
	{
	}

	virtual nn_int fan_in_size() const
	{
		return out_size();
	}

	virtual nn_int fan_out_size() const
	{
		return out_size();
	}

	virtual void set_task_count(nn_int task_count)
	{
		m_task_count = task_count;
	}

	virtual void set_batch_size(nn_int batch_size)
	{
	}

	/*
		input: input of this layer
	*/
	virtual void forw_prop(const varray &input) = 0;

	/*
		next_wd: next layer's transpose(weight) * delta
	*/
	virtual void back_prop(const varray &next_wd) = 0;

	virtual bool update_weights(nn_float batch_lr)
	{
		nn_int b_sz = m_b.size();
		nn_int w_sz = m_w.size();
		if (b_sz == 0 || w_sz == 0)
		{
			return true;
		}

		auto& ts = m_task_storage[0];

//#ifdef _DEBUG
		if (!is_valid(ts.m_db) || !is_valid(ts.m_dw))
		{
			return false;
		}
//#endif

		// update weights
		fo_vv(&ts.m_db[0], b_sz, -batch_lr, &m_b[0], b_sz);
		fo_vv(&ts.m_dw[0], w_sz, -batch_lr, &m_w[0], w_sz);

		// clear task storage
		ts.m_dw.make_zero();
		ts.m_db.make_zero();

		return true;

	}

};
}
#endif //__LAYER_H__
