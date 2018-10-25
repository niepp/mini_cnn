#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

#include <map>

struct  Cost
{
	int cnt;
	float cost;
};

struct Shape
{
	int iw, ih;
	int fw, fh;
	int ow, oh;
	bool operator<(const Shape &other) const
	{
		if (iw == other.iw)
		{
			if (ih == other.ih)
			{
				if (fw == other.fw)
				{
					if (fh == other.fh)
					{
						return ow < other.ow;
					}
					else
					{
						return fh < other.fh;
					}
				}
				else
				{
					return fw < other.fw;
				}
			}
			else
			{
				return ih < other.ih;
			}
		}
		else
		{
			return iw < other.iw;
		}
	}
};

std::map<Shape, Cost> g_shape_map;
int g_conv_cnt = 0;

namespace mini_cnn 
{

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
	nn_int m_filter_count;
	nn_int m_stride_w;
	nn_int m_stride_h;
	padding_type m_padding;
	active_func m_f;
	active_func m_df;

public:
	convolutional_layer(nn_int filter_w, nn_int filter_h, nn_int filter_c, nn_int filter_n, nn_int stride_w, nn_int stride_h, padding_type padding, activation_type ac_type)
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

		nn_assert(m_prev->m_out_shape.m_d == m_filter_shape.m_d);

		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int out_w = static_cast<nn_int>(::floorf(1.0f * (in_w - m_filter_shape.m_w) / m_stride_w)) + 1;
		nn_int out_h = static_cast<nn_int>(::floorf(1.0f * (in_h - m_filter_shape.m_h) / m_stride_h)) + 1;
		m_out_shape.set(out_w, out_h, m_filter_count);

		m_b.resize(m_filter_count);
		m_w.resize(m_filter_shape.m_w, m_filter_shape.m_h, m_filter_shape.m_d, m_filter_count);
	}

	virtual nn_int fan_in_size() const
	{
		return m_filter_shape.size();
	}

	virtual nn_int fan_out_size() const
	{
		return (m_filter_shape.m_w / m_stride_w) * (m_filter_shape.m_h / m_stride_h) * m_filter_count;
	}

	virtual void set_task_count(nn_int task_count)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;

		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int out_d = m_out_shape.m_d;

		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			ts.m_dw.resize(m_w.width(), m_w.height(), m_w.depth(), m_w.count());
			ts.m_db.resize(m_filter_count);
			ts.m_z.resize(out_w, out_h, out_d);
			if (!m_out_shape.is_img())
			{
				ts.m_x.resize(out_w * out_h * out_d);
			}
			else
			{
				ts.m_x.resize(out_w, out_h, out_d);
			}
			ts.m_delta.resize(out_w, out_h, out_d);
			ts.m_wd.resize(in_w, in_h, in_d);
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		varray &out_z = m_task_storage[task_idx].m_z;
		varray &out_x = m_task_storage[task_idx].m_x;

		conv_input_w(input, m_w, m_stride_w, m_stride_h, out_z);

		for (nn_int k = 0; k < m_out_shape.m_d; ++k)
		{
			for (nn_int i = 0; i < m_out_shape.m_w; ++i)
			{
				for (nn_int j = 0; j < m_out_shape.m_h; ++j)
				{
					out_z(i, j, k) += m_b(k);
				}
			}
		}

		m_f(out_z, out_x);

		if (m_next != nullptr)
		{
			m_next->forw_prop(out_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, nn_int task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];
		const varray &input = m_prev->get_output(task_idx);

		nn_int out_sz = next_wd.size();
		nn_int in_sz = input.size();

		/*
			delta = next_wd element-wise multiplication df(z)
		*/
		m_df(ts.m_z, ts.m_delta);
		for (nn_int i = 0; i < out_sz; ++i)
		{
			ts.m_delta[i] *= next_wd[i];
		}

		/*
			dw_k = conv2d(input_d, delta_k)
		*/
		conv_input_delta(input, ts.m_delta, m_stride_w, m_stride_h, ts.m_dw);

		/*
			db_k = sum(delta_k)
		*/
		for (nn_int k = 0; k < m_filter_count; ++k)
		{
			nn_float s = 0;
			for (nn_int i = 0; i < m_out_shape.m_w; ++i)
			{
				for (nn_int j = 0; j < m_out_shape.m_h; ++j)
				{
					s += ts.m_delta(i, j, k);
				}
			}
			ts.m_db(k) += s;
		}
		
		/*
			wd = conv(w, delta)
		*/
		conv_delta_w(ts.m_delta, m_w, m_stride_w, m_stride_h, ts.m_wd);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	static void conv_input_w(const varray &in_img, const varray &filters, nn_int stride_w, nn_int stride_h, varray &out_img)
	{
		nn_int in_w = in_img.width();
		nn_int in_h = in_img.height();
		nn_int in_d = in_img.depth();

		nn_int filter_count = filters.count();
		nn_int filter_w = filters.width();
		nn_int filter_h = filters.height();
		nn_int filter_d = filters.depth();

		nn_int w = out_img.width();
		nn_int h = out_img.height();
		nn_int d = out_img.depth();

		nn_assert(in_img.check_dim(3));
		nn_assert(filters.check_dim(4));
		nn_assert(out_img.check_dim(3));

		nn_assert(in_d == filter_d);
		nn_assert(d == filter_count);

		out_img.make_zero();

		for (nn_int k = 0; k < filter_count; ++k)
		{
			for (nn_int c = 0; c < in_d; ++c)
			{
				conv_2d<false>(&in_img(0, 0, c), in_w, in_h
					, &filters(0, 0, c, k), filter_w, filter_h
					, stride_w, stride_h
					, &out_img(0, 0, k), w, h);
			}
		}
	}

	static void conv_input_delta(const varray &in_img, const varray &delta, nn_int stride_w, nn_int stride_h, varray &dw)
	{
		nn_int in_w = in_img.width();
		nn_int in_h = in_img.height();
		nn_int in_d = in_img.depth();

		nn_int delta_w = delta.width();
		nn_int delta_h = delta.height();
		nn_int delta_d = delta.depth();

		nn_int w = dw.width();
		nn_int h = dw.height();
		nn_int d = dw.depth();
		nn_int n = dw.count();

		nn_assert(in_img.check_dim(3));
		nn_assert(dw.check_dim(4));
		nn_assert(in_d == d);
		nn_assert(n == delta_d);

		dw.make_zero();

		for (nn_int k = 0; k < n; ++k)
		{
			for (nn_int c = 0; c < d; ++c)
			{
				conv_2d<false>(&in_img(0, 0, c), in_w, in_h
					, &delta(0, 0, k), delta_w, delta_h
					, stride_w, stride_h
					, &dw(0, 0, c, k), w, h);
			}
		}
	}

	static void conv_delta_w(const varray &delta, const varray &filters, nn_int stride_w, nn_int stride_h, varray &ret)
	{
		nn_int delta_w = delta.width();
		nn_int delta_h = delta.height();
		nn_int delta_d = delta.depth();

		nn_int filter_count = filters.count();
		nn_int filter_w = filters.width();
		nn_int filter_h = filters.height();
		nn_int filter_d = filters.depth();

		nn_int w = ret.width();
		nn_int h = ret.height();
		nn_int d = ret.depth();

		nn_assert(delta.check_dim(3));
		nn_assert(filters.check_dim(4));
		nn_assert(ret.check_dim(3));

		nn_assert(delta_d == filter_count);

		nn_assert(d == filter_d);

		ret.make_zero();

		for (nn_int c = 0; c < d; ++c)
		{
			for (nn_int k = 0; k < filter_count; ++k)
			{
				conv_2d<true>(&delta(0, 0, k), delta_w, delta_h
					, &filters(0, 0, c, k), filter_w, filter_h
					, stride_w, stride_h
					, &ret(0, 0, c), w, h);
			}
		}
	}

	/*
		common 2d convolution
	*/
	template<bool filter_flip>
	static void conv_2d(const nn_float *img, nn_int iw, nn_int ih
		, const nn_float *filter, nn_int fw, nn_int fh
		, nn_int stride_w, nn_int stride_h
		, nn_float *out, nn_int ow, nn_int oh)
	{

		//int t0 = get_now();
		/*if (fw == 3 && fh == 3)
		{
		conv_2d_3x3<filter_flip>(img, iw, ih, filter, fw, fh, stride_w, stride_h, out, ow, oh);
		}
		else*/
		{
			for (nn_int i = 0; i < ow; ++i)
			{
				for (nn_int j = 0; j < oh; ++j)
				{
					nn_int start_w = i * stride_w;
					nn_int start_h = j * stride_h;
					nn_float s = out[i + j * ow];
					for (nn_int u = 0; u < fw; ++u)
					{
						nn_int x = filter_flip ? start_w - u : start_w + u;
						if (x < 0 || x >= iw)
						{
							continue;
						}
						for (nn_int v = 0; v < fh; ++v)
						{
							nn_int y = filter_flip ? start_h - v : start_h + v;
							if (y < 0 || y >= ih)
							{
								continue;
							}
							s += img[x + y * iw] * filter[u + v * fw];
						}
					}
					out[i + j * ow] = s;
				}
			}
		}
		//int t1 = get_now();

		//Shape shape;
		//shape.iw = iw;
		//shape.ih = ih;
		//shape.fw = fw;
		//shape.fh = fh;
		//shape.ow = ow;
		//shape.oh = oh;
		//++g_shape_map[shape].cnt;
		//g_shape_map[shape].cost += (t1 - t0) * 0.001f;
		//++g_conv_cnt;

	}

	template<bool filter_flip>
	static void conv_2d_3x3(const nn_float *img, nn_int iw, nn_int ih
		, const nn_float *filter, nn_int fw, nn_int fh
		, nn_int stride_w, nn_int stride_h
		, nn_float *out, nn_int ow, nn_int oh)
	{
		if (!filter_flip)
		{
			for (nn_int i = 0; i < ow; ++i)
			{
				for (nn_int j = 0; j < oh; ++j)
				{
					nn_int start_w = i * stride_w;
					nn_int start_h = j * stride_h;
					nn_float s = out[i + j * ow];

					nn_int x = start_w;
					nn_int y = start_h;
					s += ((x >= iw || y >= ih) ? 0 : img[x + y * iw]) * filter[0];
					s += ((x + 1 >= iw || y >= ih) ? 0 : img[x + 1 + y * iw]) * filter[1];
					s += ((x + 2 >= iw || y >= ih) ? 0 : img[x + 2 + y * iw]) * filter[2];

					s += ((x >= iw || y + 1 >= ih) ? 0 : img[x + (y + 1) * iw]) * filter[3];
					s += ((x + 1 >= iw || y + 1 >= ih) ? 0 : img[x + 1 + (y + 1) * iw]) * filter[4];
					s += ((x + 2 >= iw || y + 1 >= ih) ? 0 : img[x + 2 + (y + 1) * iw]) * filter[5];

					s += ((x >= iw || y + 2 >= ih) ? 0 : img[x + (y + 2) * iw]) * filter[6];
					s += ((x + 1 >= iw || y + 2 >= ih) ? 0 : img[x + 1 + (y + 2) * iw]) * filter[7];
					s += ((x + 2 >= iw || y + 2 >= ih) ? 0 : img[x + 2 + (y + 2) * iw]) * filter[8];

					out[i + j * ow] = s;
				}
			}
		}
		else
		{
			for (nn_int i = 0; i < ow; ++i)
			{
				for (nn_int j = 0; j < oh; ++j)
				{
					nn_int start_w = i * stride_w;
					nn_int start_h = j * stride_h;
					nn_float s = out[i + j * ow];

					nn_int x = start_w;
					nn_int y = start_h;
					s += ((x >= iw || y >= ih) ? 0 : img[x + y * iw]) * filter[0];
					s += ((x - 1 < 0 || x - 1 >= iw || y >= ih) ? 0 : img[x - 1 + y * iw]) * filter[1];
					s += ((x - 2 < 0 || x - 2 >= iw || y >= ih) ? 0 : img[x - 2 + y * iw]) * filter[2];

					s += ((y - 1 < 0 || x >= iw || y - 1 >= ih) ? 0 : img[x + (y - 1) * iw]) * filter[3];
					s += ((x - 1 < 0 || y - 1 < 0 || x - 1 >= iw || y - 1 >= ih) ? 0 : img[x - 1 + (y - 1) * iw]) * filter[4];
					s += ((x - 2 < 0 || y - 1 < 0 || x - 2 >= iw || y - 1 >= ih) ? 0 : img[x - 2 + (y - 1) * iw]) * filter[5];

					s += ((y - 2 < 0 || x >= iw || y - 2 >= ih) ? 0 : img[x + (y - 2) * iw]) * filter[6];
					s += ((x - 1 < 0 || y - 2 < 0 || x - 1 >= iw || y - 2 >= ih) ? 0 : img[x - 1 + (y - 2) * iw]) * filter[7];
					s += ((x - 2 < 0 || y - 2 < 0 || x - 2 >= iw || y - 2 >= ih) ? 0 : img[x - 2 + (y - 2) * iw]) * filter[8];

					out[i + j * ow] = s;
				}
			}
		}
	}

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

