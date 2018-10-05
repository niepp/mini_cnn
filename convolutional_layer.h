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

protected:
	void calc_outsize(const shape3d &input, const shape3d &filter, int filter_count, shape3d &output)
	{
		int_t in_w = input.m_w;
		int_t in_h = input.m_h;
		int_t out_w = static_cast<int_t>(::floorf(1.0f * (in_w - filter.m_w) / m_stride_w)) + 1;
		int_t out_h = static_cast<int_t>(::floorf(1.0f * (in_h - filter.m_h) / m_stride_h)) + 1;
		output.set(out_w, out_h, filter_count);
	}

public:
	convolutional_layer(int_t filter_w, int_t filter_h, int_t filter_c, int_t filter_n, int_t stride_w, int_t stride_h, padding_type padding, activation_type ac_type)
		: layer_base()
		, m_filter_shape(filter_w, filter_h, filter_c)
		, m_stride_w(stride_w), m_stride_h(stride_h), m_padding(padding)
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
		calc_outsize(m_prev->m_out_shape, m_filter_shape, m_filter_count, m_out_shape);
		m_b.resize(m_filter_count);
		m_w.resize(m_filter_shape.m_w, m_filter_shape.m_h, m_filter_shape.m_d, m_filter_count);
	}

	virtual void set_task_count(int_t task_count)
	{
		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.resize(m_w.width(), m_w.height(), m_w.depth(), m_w.count());
			ts.m_db.resize(m_filter_count);
			ts.m_z.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
			ts.m_x.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
			ts.m_delta.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
			ts.m_wd.resize(m_out_shape.m_w, m_out_shape.m_h, m_out_shape.m_d);
		}
	}

	virtual const varray& forw_prop(const varray &input, int_t task_idx)
	{
		varray &out_z = m_task_storage[task_idx].m_z;
		varray &out_x = m_task_storage[task_idx].m_x;
		conv(input, m_w, m_stride_w, m_stride_h, out_z);
		m_f(out_z, out_x);

		if (m_next != nullptr)
		{
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
			ts.m_delta(i) *= next_wd(i);
		}

		// conv_back();

		return next_wd;
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

	static void conv(const varray &img, const varray &filters, int_t stride_w, int_t stride_h, varray &ret)
	{
		int_t filter_count = filters.count();
		int_t filter_w = filters.width();
		int_t filter_h = filters.height();
		int_t filter_d = filters.depth();

		nn_assert(img.width() >= filters.width()
			&& img.height() >= filters.height()
			&& img.depth() == filters.depth());

		nn_assert(ret.depth() == filter_count);

		for (int_t i = 0; i < filter_w; ++i)
		{
			for (int_t j = 0; j < filter_h; ++j)
			{
				int_t start_w = i * stride_w;
				int_t start_h = j * stride_h;
				for (int_t k = 0; k < filter_count; ++k)
				{
					float_t s = conv_by_local(img, start_w, start_h, filters, k);
					ret(i, j, k) = s;
				}
			}
		}
	}

	static void conv_back(const varray &delta_img, const varray &filters, int_t stride_w, int_t stride_h, varray &ret)
	{
		int_t img_w = delta_img.width();
		int_t img_h = delta_img.height();
		int_t img_d = delta_img.depth();

		int_t filter_count = filters.count();
		int_t filter_w = filters.width();
		int_t filter_h = filters.height();
		int_t filter_d = filters.depth();

		assert(delta_img.width() >= filter_w
			&& delta_img.height() >= filter_h
			&& delta_img.depth() == filter_d);

		int_t cnt = ret.size();

		for (int_t k = 0; k < cnt; ++k)
		{			
			for (int_t i = 0; i < filter_w; ++i)
			{
				for (int_t j = 0; j < filter_h; ++j)
				{
					int_t start_w = i * stride_w;
					int_t start_h = j * stride_h;
					for (int_t c = 0; c < filter_d; ++c)
					{
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
								if (y < 0 || y >= img_w)
								{
									continue;
								}
								s += delta_img(x, y, c) * filters(u, v, k);
							}
						}
						ret(i, j, c, k) = s;
					}
				}
			}
		}
	}



};
}
#endif //__CONVOLUTIONAL_LAYER_H__

