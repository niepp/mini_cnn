#ifndef __BATCH_NORMALIZATION_LAYER_H__
#define __BATCH_NORMALIZATION_LAYER_H__

namespace mini_cnn 
{

class batch_normalization_layer : public layer_base
{
protected:
	varray m_batch_mean;
	varray m_batch_var;

	varray m_total_mean;
	varray m_total_var;

public:
	batch_normalization_layer()
		: layer_base()
	{
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);
		m_out_shape = m_prev->m_out_shape;

		nn_int sz = m_out_shape.size();
		m_b.resize(sz); // beta
		m_w.resize(sz); // gamma
	}

	virtual void set_task_count(nn_int task_count)
	{
		layer_base::set_task_count(task_count);

		nn_int sz = m_out_shape.size();
		m_task_storage.resize(1);

		// beta & gamma is shared in a batch
		auto& ts = m_task_storage[0];
		{
			ts.m_dw.resize(sz);
			ts.m_db.resize(sz);
		}

		m_batch_mean.resize(sz);
		m_batch_var.resize(sz);

		m_total_mean.resize(sz);
		m_total_var.resize(sz);

	}

	virtual void set_batch_size(nn_int batch_size)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int out_d = m_out_shape.m_d;

		nn_assert(in_w == out_w);
		nn_assert(in_h == out_h);
		nn_assert(in_d == out_d);

		nn_int out_sz = m_out_shape.size();
		m_z_vec.resize(out_sz, 1, 1, batch_size); // used for norm_x
		m_x_vec.resize(out_sz, 1, 1, batch_size); // bn output
		if (m_prev->m_out_shape.is_img())
		{
			m_wd_vec.resize(in_w, in_h, in_d, batch_size);
		}
		else
		{
			m_wd_vec.resize(in_w * in_h * in_d, 1, 1, batch_size);
		}
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_int batch_size = input_batch.count();

		bn(input_batch, m_x_vec);

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}
	}

	virtual void back_prop(const varray &next_wd)
	{
		nn_int batch_size = next_wd.count();
		nn_int sz = m_out_shape.size();

		varray &gamma = m_task_storage[0].m_dw;
		varray &beta = m_task_storage[0].m_db;

		for (int b = 0; b < batch_size; ++b)
		{
			const nn_float *vec_next_wd = next_wd.data(b);
			nn_float *nn_restrict z = m_z_vec.data(b);
			nn_float *nn_restrict x = m_x_vec.data(b);
			
			nn_float *nn_restrict wd = m_wd_vec.data(b);

			/*
				prev delta := w * delta กั df(z)
			*/
			for (nn_int i = 0; i < sz; ++i)
			{
				wd[i] = vec_next_wd[i] * x[i];
				beta[i] = vec_next_wd[i] * x[i];
				gamma[i] = beta[i] * z[i];
			}

		}

		m_prev->back_prop(m_wd_vec);

	}

	void bn(const varray &input_batch, varray &out_batch)
	{
		nn_assert(input_batch.count() == out_batch.count());
		m_batch_mean.make_zero();
		m_batch_var.make_zero();
		nn_float *vec_mean = m_batch_mean.data();
		nn_float *vec_var = m_batch_var.data();
		nn_int sz = input_batch.img_size();
		nn_int batch_size = input_batch.count();

		// mean & variance
		// var[x] = E[x^2] - E[x]^2
		for (nn_int b = 0; b < batch_size; ++b)
		{
			const nn_float *nn_restrict input = input_batch.data(b);
			for (nn_int i = 0; i < sz; ++i)
			{
				vec_mean[i] += input[i];
				vec_var[i] += input[i] * input[i];
			}
		}

		nn_float inv_batch_size = cOne / batch_size;
		for (nn_int i = 0; i < sz; ++i)
		{
			vec_mean[i] *= inv_batch_size;
			vec_var[i] *= inv_batch_size;
		}

		for (nn_int i = 0; i < sz; ++i)
		{
			vec_var[i] -= vec_mean[i] * vec_mean[i];
		}

		for (nn_int b = 0; b < batch_size; ++b)
		{
			const nn_float *nn_restrict input = input_batch.data(b);
			nn_float *nn_restrict out = out_batch.data(b);
			nn_float *nn_restrict z = m_z_vec.data(b);
			for (nn_int i = 0; i < sz; ++i)
			{
				nn_float norm_x = (input[i] - vec_mean[i]) * fast_inv_sqrt(vec_var[i] + cEpsilon);
				z[i] = norm_x;
				out[i] = m_w[i] * norm_x + m_b[i];
			}
		}

	}

};
}
#endif //__BATCH_NORMALIZATION_LAYER_H__

