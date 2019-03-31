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

	varray m_dJ_dvar;
	varray m_dJ_dmean;

	bool m_total_init;
	nn_float m_decay;

public:
	batch_normalization_layer(nn_float decay = (nn_float)0.99)
		: layer_base(), m_decay(decay)
	{
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);
		m_out_shape = m_prev->m_out_shape;

		nn_int sz = m_out_shape.size();
		m_b.resize(sz); // beta
		m_w.resize(sz); // gamma

		m_batch_mean.resize(sz);
		m_batch_var.resize(sz);

		m_total_mean.resize(sz);
		m_total_var.resize(sz);
		m_dJ_dvar.resize(sz);
		m_dJ_dmean.resize(sz);
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

	virtual void set_phase_type(phase_type phase)
	{
		layer_base::set_phase_type(phase);
		m_total_init = false;
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_int batch_size = input_batch.count();
		if (m_phase_type == phase_type::eTrain)
		{
			bn(input_batch, m_x_vec);
		}
		else if (m_phase_type == phase_type::eGradientCheck)
		{

		}
		else if (m_phase_type == phase_type::eTest)
		{
			nn_int sz = input_batch.img_size();
			for (nn_int b = 0; b < batch_size; ++b)
			{
				const nn_float *nn_restrict input = input_batch.data(b);
				nn_float *nn_restrict out = m_x_vec.data(b);
				for (nn_int i = 0; i < sz; ++i)
				{
					nn_float norm_x = (input[i] - m_total_mean[i]) * fast_inv_sqrt(m_total_var[i] + cEpsilon);
					out[i] = m_w[i] * norm_x + m_b[i];
				}
			}
		}

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}
	}

	virtual void back_prop(const varray &next_wd)
	{
		nn_int batch_size = next_wd.count();
		nn_int sz = m_out_shape.size();

		varray &d_gamma = m_task_storage[0].m_dw;
		varray &d_beta = m_task_storage[0].m_db;

		nn_float *vec_mean = m_batch_mean.data();
		nn_float *vec_var = m_batch_var.data();

		const varray &input_batch = m_prev->get_output();

		m_dJ_dvar.make_zero();
		m_dJ_dmean.make_zero();
		m_wd_vec.make_zero();

		for (int b = 0; b < batch_size; ++b)
		{
			const nn_float *nn_restrict input = input_batch.data(b);
			const nn_float *vec_next_wd = next_wd.data(b);
			for (nn_int i = 0; i < sz; ++i)
			{
				m_dJ_dvar[i] += (nn_float)(-0.5) * (vec_next_wd[i] * m_w[i]) * (input[i] - vec_mean[i]) * fast_inv_sqrt(vec_var[i] + cEpsilon) / (vec_var[i] + cEpsilon);
			}

			for (nn_int i = 0; i < sz; ++i)
			{
				m_dJ_dmean[i] += -(vec_next_wd[i] * m_w[i]) * fast_inv_sqrt(vec_var[i] + cEpsilon);
				m_dJ_dmean[i] -= m_dJ_dvar[i] * (nn_float)2.0 * (input[i] - vec_mean[i]) / batch_size;
			}
		}
		
		for (int b = 0; b < batch_size; ++b)
		{
			const nn_float *nn_restrict input = input_batch.data(b);
			const nn_float *vec_next_wd = next_wd.data(b);
			nn_float *nn_restrict z = m_z_vec.data(b);
			nn_float *nn_restrict x = m_x_vec.data(b);
			nn_float *nn_restrict wd = m_wd_vec.data(b);

			/*
				prev delta := w * delta กั df(z)
			*/
			for (nn_int i = 0; i < sz; ++i)
			{
				//wd[i] = vec_next_wd[i] * x[i];
				d_beta[i] += vec_next_wd[i];
				d_gamma[i] += vec_next_wd[i] * z[i];
			}

			for (nn_int i = 0; i < sz; ++i)
			{
				// dJ/d(x^)
				wd[i] = vec_next_wd[i] * m_w[i];
				wd[i] *= fast_inv_sqrt(vec_var[i] + cEpsilon);
				wd[i] += m_dJ_dvar[i] * (nn_float)2.0 * (input[i] - vec_mean[i]) / batch_size;
				wd[i] += m_dJ_dmean[i] / batch_size;
			}

			//for (nn_int i = 0; i < sz; ++i)
			//{
			//	// dJ/d(x^)
			//	wd[i] = vec_next_wd[i] * m_w[i];
			//	nn_float inv_v = fast_inv_sqrt(vec_var[i] + cEpsilon);
			//	nn_float di = (input[i] - m_batch_mean[i]);
			//	wd[i] *= inv_v * (cOne - cOne / batch_size) + di * inv_v * inv_v * inv_v * ((nn_float)2.0 / batch_size) * di * (cOne - input[i] / batch_size);
			//}
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

		if (!m_total_init)
		{
			m_total_init = true;
			m_total_mean.copy(m_batch_mean);
			m_total_var.copy(m_batch_var);
		}
		else
		{
			nn_float var_nobias = (batch_size > 1) ? batch_size / (batch_size - 1) : cOne;
			for (nn_int i = 0; i < sz; ++i)
			{
				m_total_mean[i] = var_nobias * m_total_mean[i] + (cOne - var_nobias) * vec_mean[i];
				m_total_var[i] = var_nobias * m_total_var[i] + (cOne - var_nobias) * vec_var[i];
			}
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

