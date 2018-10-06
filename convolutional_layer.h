#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

namespace mini_cnn 
{

enum class padding_type
{
	valid, // only use valid pixels of input
	same   // padding zero around input to keep image size
};

/*
for the l-th Conv layer:
	(l)     (l-1)      (l)    (l)
   Z	=  X       *  W    + B
    (l)      (l)
   X    = f(Z   )
*/
class convolutional_layer : public layer_base
{

protected:
	shape3d m_filter_shape;
	int_t m_filter_count;
	int_t m_stride_w;
	int_t m_stride_h;
	padding_type m_padding;
	active_func m_f;
	active_func m_df;

public:
	convolutional_layer(int_t filter_w, int_t filter_h, int_t filter_c, int_t filter_n, int_t stride_w, int_t stride_h, padding_type padding, activation_type ac_type)
		: layer_base()
		, m_filter_shape(filter_w, filter_h, filter_c)
		, m_filter_count(filter_n), m_stride_w(stride_w), m_stride_h(stride_h), m_padding(padding)
	{
		switch (ac_type)
		{
		case activation_type::eSigmod:
			m_f = sigmoid;
			m_df = deriv_sigmoid;
			break;
		case activation_type::eRelu:
			m_f = relu;
			m_df = deriv_relu;
			break;
		case activation_type::eSoftMax:
			m_f = softmax;
			m_df = nullptr;
			break;
		default:
			break;
		}
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);
		
		nn_assert(m_prev->m_out_shape.is_img());

		int_t in_w = m_prev->m_out_shape.m_w;
		int_t in_h = m_prev->m_out_shape.m_h;
		int_t out_w = static_cast<int_t>(::floorf(1.0f * (in_w - m_filter_shape.m_w) / m_stride_w)) + 1;
		int_t out_h = static_cast<int_t>(::floorf(1.0f * (in_h - m_filter_shape.m_h) / m_stride_h)) + 1;
		m_out_shape.set(out_w, out_h, m_filter_count);

		m_b.resize(m_filter_count);
		m_w.resize(m_filter_shape.m_w, m_filter_shape.m_h, m_filter_shape.m_d, m_filter_count);
	}

	virtual void set_task_count(int_t task_count)
	{
		int_t in_w = m_prev->m_out_shape.m_w;
		int_t in_h = m_prev->m_out_shape.m_h;
		int_t in_d = m_prev->m_out_shape.m_d;

		int_t out_w = m_out_shape.m_w;
		int_t out_h = m_out_shape.m_h;
		int_t out_d = m_out_shape.m_d;

		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.resize(m_w.width(), m_w.height(), m_w.depth(), m_w.count());
			ts.m_db.resize(m_filter_count);
			ts.m_z.resize(out_w, out_h, out_d);
			ts.m_x.resize(out_w, out_h, out_d);
			ts.m_delta.resize(out_w, out_h, out_d);
			ts.m_wd.resize(in_w, in_h, in_d);
		}
	}

	virtual const varray& forw_prop(const varray &input, int_t task_idx)
	{
		varray &out_z = m_task_storage[task_idx].m_z;
		varray &out_x = m_task_storage[task_idx].m_x;

		conv3d(input, m_w, m_stride_w, m_stride_h, out_z);
		m_f(out_z, out_x);

		if (m_next != nullptr)
		{
			if (!m_next->m_out_shape.is_img())
			{
				out_x.reshape(out_x.size());
			}
			return m_next->forw_prop(out_x, task_idx);
		}
		return out_x;
	}

	virtual const varray& back_prop(const varray &next_wd, int_t task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];
		const varray &input = m_prev->get_output(task_idx);

		int_t out_sz = next_wd.size();
		int_t in_sz = input.size();

		m_df(ts.m_z, ts.m_delta);
		for (int_t i = 0; i < out_sz; ++i)
		{
			ts.m_delta[i] *= next_wd[i];
		}

		/*
			dw_k = conv2d(input_d, delta_k)
		*/
		conv2d(input, ts.m_delta, m_stride_w, m_stride_h, ts.m_dw);

		/*
			db_k = sum(delta_k)
		*/
		for (int_t k = 0; k < m_filter_count; ++k)
		{
			float_t s = 0;
			for (int_t i = 0; i < m_out_shape.m_w; ++i)
			{
				for (int_t j = 0; j < m_out_shape.m_h; ++j)
				{
					s += ts.m_delta(i, j, k);
				}
			}
			ts.m_db(k) += s;
		}

		conv_back(ts.m_delta, ts.m_dw, m_stride_w, m_stride_h, ts.m_wd);

		return m_prev->back_prop(ts.m_wd, task_idx);
	}

private:
	static float_t conv_by_local(const varray &img, int_t start_w, int_t start_h, const varray &filters, int_t filter_idx)
	{
		int_t img_w = img.width();
		int_t img_h = img.height();
		int_t img_d = img.depth();
		int_t filter_w = filters.width();
		int_t filter_h = filters.height();
		int_t filter_d = filters.depth();

		float_t s = 0;
		for (int_t u = 0; u < filter_w; ++u)
		{
			int_t x = start_w + u;
			if (x < 0 || x >= img_w)
			{
				continue;
			}
			for (int_t v = 0; v < filter_h; ++v)
			{
				int_t y = start_h + v;
				if (y < 0 || y >= img_h)
				{
					continue;
				}
				for (int_t c = 0; c < filter_d; ++c)
				{
					s += img(x, y, c) * filters(u, v, c, filter_idx);
				}
			}
		}
		return s;
	}

	static void conv3d(const varray &in_img, const varray &filters, int_t stride_w, int_t stride_h, varray &out_img)
	{
		int_t filter_count = filters.count();
		int_t filter_w = filters.width();
		int_t filter_h = filters.height();
		int_t filter_d = filters.depth();

		int_t out_w = out_img.width();
		int_t out_h = out_img.height();
		int_t out_d = out_img.depth();

		nn_assert(in_img.check_dim(3));
		nn_assert(filters.check_dim(4));
		nn_assert(out_img.check_dim(3));

		nn_assert(in_img.depth() == filters.depth());

		nn_assert(out_d == filter_count);

		for (int_t i = 0; i < out_w; ++i)
		{
			for (int_t j = 0; j < out_h; ++j)
			{
				int_t start_w = i * stride_w;
				int_t start_h = j * stride_h;
				for (int_t k = 0; k < filter_count; ++k)
				{
					float_t s = conv_by_local(in_img, start_w, start_h, filters, k);
					out_img(i, j, k) = s;
				}
			}
		}
	}

	static void conv2d(const varray &in_img, const varray &delta, int_t stride_w, int_t stride_h, varray &dw)
	{
		int_t in_w = in_img.width();
		int_t in_h = in_img.height();
		int_t in_d = in_img.depth();

		int_t delta_w = delta.width();
		int_t delta_h = delta.height();
		int_t delta_d = delta.depth();

		int_t w = dw.width();
		int_t h = dw.height();
		int_t d = dw.depth();
		int_t n = dw.count();

		nn_assert(in_img.check_dim(3));
		nn_assert(dw.check_dim(4));
		nn_assert(in_d == d);
		nn_assert(n == delta_d);

		for (int_t k = 0; k < n; ++k)
		{
			for (int_t i = 0; i < w; ++i)
			{
				for (int_t j = 0; j < h; ++j)
				{
					int_t start_w = i * stride_w;
					int_t start_h = j * stride_h;
					for (int_t c = 0; c < d; ++c)
					{
						float_t s = 0;
						for (int_t u = 0; u < delta_w; ++u)
						{
							int_t x = start_w + u;
							if (x < 0 || x >= in_w)
							{
								continue;
							}
							for (int_t v = 0; v < delta_h; ++v)
							{
								int_t y = start_h + v;
								if (y < 0 || y >= in_h)
								{
									continue;
								}
								s += in_img(x, y, c) * delta(u, v, k);
							}
						}
						dw(i, j, c, k) += s;
					}
				}
			}
		}
	}

	static void conv_back(const varray &delta, const varray &filters, int_t stride_w, int_t stride_h, varray &ret)
	{
		int_t delta_w = delta.width();
		int_t delta_h = delta.height();
		int_t delta_d = delta.depth();

		int_t filter_count = filters.count();
		int_t filter_w = filters.width();
		int_t filter_h = filters.height();
		int_t filter_d = filters.depth();

		int_t w = ret.width();
		int_t h = ret.height();
		int_t d = ret.depth();

		nn_assert(delta.check_dim(3));
		nn_assert(filters.check_dim(4));
		nn_assert(ret.check_dim(3));

		nn_assert(delta_d == filter_count);

		nn_assert(d == filter_d);

		for (int_t c = 0; c < d; ++c)
		{
			for (int_t i = 0; i < w; ++i)
			{
				for (int_t j = 0; j < h; ++j)
				{
					int_t start_w = i * stride_w;
					int_t start_h = j * stride_h;
					float_t s = 0;
					for (int_t k = 0; k < filter_count; ++k)
					{
						for (int_t u = 0; u < filter_w; ++u)
						{
							int_t x = start_w - u;
							if (x < 0 || x >= delta_w)
							{
								continue;
							}
							for (int_t v = 0; v < filter_h; ++v)
							{
								int_t y = start_h - v;
								if (y < 0 || y >= delta_h)
								{
									continue;
								}
								s += delta(x, y, k) * filters(u, v, c, k);
							}
						}
					}
					ret(i, j, c) = s;
				}
			}
		}
	}

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

