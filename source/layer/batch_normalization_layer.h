#ifndef __BATCH_NORMALIZATION_LAYER_H__
#define __BATCH_NORMALIZATION_LAYER_H__

#include <fstream>

namespace mini_cnn
{

class batch_normalization_layer : public layer_base
{
protected:
	varray m_batch_mean;
	varray m_batch_var;

	varray m_total_mean;
	varray m_total_var;

	varray m_dJ_dxhat;
	varray m_dJ_dxhat2;
	varray m_dJ_dxhat3;

	bool m_total_init;
	nn_float m_decay;
	nn_float m_epsilon;

public:
	batch_normalization_layer(nn_float decay = (nn_float)0.99, nn_float epsilon = (nn_float)0.0001)
		: layer_base(), m_decay(decay), m_epsilon(epsilon)
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

	}

	virtual void set_task_count(nn_int task_count)
	{
		layer_base::set_task_count(task_count);

		nn_int sz = m_out_shape.size();

		// beta & gamma is shared in a batch
		m_task_storage.resize(1);
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

		nn_int sz = m_out_shape.size();

		if (!m_out_shape.is_img())
		{
			m_z_vec.resize(sz, 1, 1, batch_size); // used for norm_x
			m_x_vec.resize(sz, 1, 1, batch_size); // bn output
		}
		else
		{
			m_z_vec.resize(out_w, out_h, out_d, batch_size); // used for norm_x
			m_x_vec.resize(out_w, out_h, out_d, batch_size); // bn output
		}

		if (m_prev->m_out_shape.is_img())
		{
			m_wd_vec.resize(in_w, in_h, in_d, batch_size);
		}
		else
		{
			m_wd_vec.resize(in_w * in_h * in_d, 1, 1, batch_size);
		}

		m_dJ_dxhat.resize(sz, 1, 1, batch_size);
		m_dJ_dxhat2.resize(sz);
		m_dJ_dxhat3.resize(sz);

	}

	virtual void set_phase_type(phase_type phase)
	{
		layer_base::set_phase_type(phase);
		m_total_init = false;
	}

	virtual void load_weights(std::fstream &fread)
	{
		nn_int wsize = 0;
		fread.read(reinterpret_cast<char*>(&wsize), sizeof(nn_int));
		nn_assert(wsize == m_w.size());
		fread.read(reinterpret_cast<char*>(m_w.data()), wsize * sizeof(nn_float));

		nn_int bsize = 0;
		fread.read(reinterpret_cast<char*>(&bsize), sizeof(nn_int));
		nn_assert(bsize == m_b.size());
		fread.read(reinterpret_cast<char*>(m_b.data()), bsize * sizeof(nn_float));

		// load mean & var of train set
		nn_int tm_size = 0;
		fread.read(reinterpret_cast<char*>(&tm_size), sizeof(nn_int));
		nn_assert(tm_size == m_total_mean.size());
		fread.read(reinterpret_cast<char*>(m_total_mean.data()), tm_size * sizeof(nn_float));

		nn_int tv_size = 0;
		fread.read(reinterpret_cast<char*>(&tv_size), sizeof(nn_int));
		nn_assert(tv_size == m_total_var.size());
		fread.read(reinterpret_cast<char*>(m_total_var.data()), tv_size * sizeof(nn_float));
	}

	virtual void save_weights(std::fstream &fwrite)
	{
		nn_int wsize = m_w.size();
		fwrite.write(reinterpret_cast<char*>(&wsize), sizeof(nn_int));
		fwrite.write(reinterpret_cast<char*>(m_w.data()), wsize * sizeof(nn_float));

		nn_int bsize = m_b.size();
		fwrite.write(reinterpret_cast<char*>(&bsize), sizeof(nn_int));
		fwrite.write(reinterpret_cast<char*>(m_b.data()), bsize * sizeof(nn_float));

		// save mean & var of train set
		nn_int tm_size = m_total_mean.size();
		fwrite.write(reinterpret_cast<char*>(&tm_size), sizeof(nn_int));
		fwrite.write(reinterpret_cast<char*>(m_total_mean.data()), tm_size * sizeof(nn_float));

		nn_int tv_size = m_total_var.size();
		fwrite.write(reinterpret_cast<char*>(&tv_size), sizeof(nn_int));
		fwrite.write(reinterpret_cast<char*>(m_total_var.data()), tv_size * sizeof(nn_float));
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_int batch_size = input_batch.count();
		if (m_phase_type == phase_type::eTrain)
		{
			batch_norm(input_batch, m_x_vec);
		}
		else if (m_phase_type == phase_type::eGradientCheck)
		{
			batch_norm(input_batch, m_x_vec);
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
					nn_float x_hat = (input[i] - m_total_mean[i]) * fast_inv_sqrt(m_total_var[i] + m_epsilon);
					out[i] = m_w[i] * x_hat + m_b[i];
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

		// [batch normalization backprop] https://kevinzakka.github.io/2016/09/14/batch_normalization/
		m_dJ_dxhat.make_zero();
		m_dJ_dxhat2.make_zero();
		m_dJ_dxhat3.make_zero();

		m_wd_vec.make_zero();

		// dJ/dx^
		for (int b = 0; b < batch_size; ++b)
		{
			nn_float *vec_dJ_dxhat = m_dJ_dxhat.data(b);
			const nn_float *vec_next_wd = next_wd.data(b);
			for (nn_int i = 0; i < sz; ++i)
			{
				vec_dJ_dxhat[i] = vec_next_wd[i] * m_w[i];
			}
		}

		for (int b = 0; b < batch_size; ++b)
		{
			const nn_float *vec_dJ_dxhat = m_dJ_dxhat.data(b);
			for (nn_int i = 0; i < sz; ++i)
			{
				m_dJ_dxhat2[i] += vec_dJ_dxhat[i];
			}
		}

		for (int b = 0; b < batch_size; ++b)
		{
			const nn_float *vec_dJ_dxhat = m_dJ_dxhat.data(b);
			const nn_float *xhat = m_z_vec.data(b);
			nn_float *vec_dJ_dxhat3 = m_dJ_dxhat3.data(0);
			for (nn_int i = 0; i < sz; ++i)
			{
				vec_dJ_dxhat3[i] += vec_dJ_dxhat[i] * xhat[i];
			}
		}

		for (int b = 0; b < batch_size; ++b)
		{
			const nn_float *vec_dJ_dxhat = m_dJ_dxhat.data(b);
			const nn_float *nn_restrict input = input_batch.data(b);
			const nn_float *vec_next_wd = next_wd.data(b);
			nn_float *nn_restrict z = m_z_vec.data(b);
			nn_float *nn_restrict x = m_x_vec.data(b);
			nn_float *nn_restrict wd = m_wd_vec.data(b);

			/*
				prev delta := w * delta ⊙ df(z)
			*/
			for (nn_int i = 0; i < sz; ++i)
			{
				d_beta[i] += vec_next_wd[i];
				d_gamma[i] += vec_next_wd[i] * z[i];
			}

			for (nn_int i = 0; i < sz; ++i)
			{
				wd[i] = (cOne / batch_size) * fast_inv_sqrt(vec_var[i] + m_epsilon) *
					(batch_size * vec_dJ_dxhat[i]
					- m_dJ_dxhat2[i]
					- z[i] * m_dJ_dxhat3[i]);
			}

		}

		m_prev->back_prop(m_wd_vec);

	}

	void batch_norm(const varray &input_batch, varray &out_batch)
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

		// use moving average to compute the Unbiased Estimation of the train set's mean & var
		// ref https://en.wikipedia.org/wiki/Moving_average
		// m_total_mean & m_total_var are computed when train and are used when test
		if (!m_total_init)
		{
			m_total_init = true;
			m_total_mean.copy(m_batch_mean);
			m_total_var.copy(m_batch_var);
		}
		else
		{
			nn_float decay = (batch_size > 1) ? m_decay : cOne;
			for (nn_int i = 0; i < sz; ++i)
			{
				m_total_mean[i] = decay * m_total_mean[i] + (cOne - decay) * vec_mean[i];
				m_total_var[i] = decay * m_total_var[i] + (cOne - decay) * vec_var[i];
			}
		}

		for (nn_int b = 0; b < batch_size; ++b)
		{
			const nn_float *nn_restrict input = input_batch.data(b);
			nn_float *nn_restrict out = out_batch.data(b);
			nn_float *nn_restrict z = m_z_vec.data(b);
			for (nn_int i = 0; i < sz; ++i)
			{
				nn_float x_hat = (input[i] - vec_mean[i]) * fast_inv_sqrt(vec_var[i] + m_epsilon);
				z[i] = x_hat;
				out[i] = m_w[i] * x_hat + m_b[i];
			}
		}

	}

};
}
#endif //__BATCH_NORMALIZATION_LAYER_H__
