#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

namespace mini_cnn 
{

class mem_block
{
public:
	nn_int w;
	nn_int h;
	nn_float *data;
	nn_int data_len;
	mem_block() : data(nullptr), data_len(0), w(0), h(0)
	{
	}
	void resize(nn_int len)
	{
		create(len);
	}
	~mem_block()
	{
		release();
	}
private:
	void create(nn_int len)
	{
		data = (nn_float*)align_malloc(len * sizeof(nn_float), nn_align_size);
		memset(data, 0, len * sizeof(nn_float));
		data_len = len;
		w = 0;
		h = 0;
	}
	void release()
	{
		if (data != nullptr)
		{
			align_free(data);
		}
		data = nullptr;
		data_len = 0;
		w = 0;
		h = 0;
	}
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
	nn_int m_filter_count;
	nn_int m_stride_w;
	nn_int m_stride_h;
	padding_type m_padding;
	active_func m_f;
	active_func m_df;

	struct conv_task_storage
	{
		mem_block m_img_block;
	};
	std::vector<conv_task_storage> m_conv_task_storage;

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

#ifdef nnGEMM
		nn_int temp_w = in_w + in_w - out_w;
		nn_int temp_h = in_h + in_h - out_h;
		nn_int img2row_size = temp_w * temp_h * m_filter_shape.m_w * m_filter_shape.m_h * m_filter_shape.m_d;
		m_conv_task_storage.resize(task_count);
		for (auto &cts : m_conv_task_storage)
		{
			cts.m_img_block.resize(img2row_size);
		}
#else
		m_conv_task_storage.resize(task_count);
		for (auto &cts : m_conv_task_storage)
		{
			cts.m_img_block.resize(in_w * in_h);
		}
#endif
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		varray &out_z = m_task_storage[task_idx].m_z;
		varray &out_x = m_task_storage[task_idx].m_x;

		mem_block &block = m_conv_task_storage[task_idx].m_img_block;

#ifdef nnGEMM		
		conv_input_w(input, block, m_w, m_stride_w, m_stride_h, out_z);
#else
		conv_input_w(input, block, m_w, m_stride_w, m_stride_h, out_z);
#endif

		for (nn_int k = 0; k < m_out_shape.m_d; ++k)
		{
			nn_float bk = m_b(k);
			for (nn_int i = 0; i < m_out_shape.m_h; ++i)
			{
				nn_float *nn_restrict vec_out_z = &out_z(0, i, k);
				for (nn_int j = 0; j < m_out_shape.m_w; ++j)
				{
					vec_out_z[j] += bk;
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

		/*
			delta := next_wd กั df(z)
		*/
		m_df(ts.m_z, ts.m_delta);

		nn_float *nn_restrict vec_delta = &ts.m_delta[0];
		const nn_float *nn_restrict vec_next_wd = &next_wd[0];
		for (nn_int i = 0; i < out_sz; ++i)
		{
			vec_delta[i] *= vec_next_wd[i];
		}

		/*
			dw_k := conv2d(input_d, delta_k)
		*/
		mem_block &block = m_conv_task_storage[task_idx].m_img_block;
#ifdef nnGEMM
		conv_input_delta(input, block, ts.m_delta, m_stride_w, m_stride_h, ts.m_dw);
#else
		conv_input_delta(input, block, ts.m_delta, m_stride_w, m_stride_h, ts.m_dw);
#endif

		/*
			db_k := sum(delta_k)
		*/
		for (nn_int k = 0; k < m_filter_count; ++k)
		{
			nn_float s = 0;
			for (nn_int i = 0; i < m_out_shape.m_h; ++i)
			{
				const nn_float *nn_restrict vec_delta_hd = &ts.m_delta(0, i, k);
				for (nn_int j = 0; j < m_out_shape.m_w; ++j)
				{
					s += vec_delta_hd[j];
				}
			}
			ts.m_db(k) += s;
		}

		/*
			wd := conv(delta, w)
		*/
		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int offset_w = input.width() - out_w;
		nn_int offset_h = input.height() - out_h;

#ifdef nnGEMM
		conv_delta_w(ts.m_delta, block, m_w, m_stride_w, m_stride_h, ts.m_wd);
#else
		conv_delta_w(ts.m_delta, block, m_w, m_stride_w, m_stride_h, ts.m_wd);
#endif
		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:

#ifdef nnGEMM
	static inline void img2row(const nn_float *img, nn_int iw, nn_int ih, nn_int channels
		, nn_int fw, nn_int fh
		, nn_int stride_iw, nn_int stride_ih
		, nn_int ow, nn_int oh
		, nn_int stride_ow, nn_int stride_oh
		, mem_block &block)
	{
		block.w = fw * fh * channels;
		block.h = ow * oh;
		nn_float *prow = block.data;
		for (nn_int i = 0; i < oh; ++i)
		{
			for (nn_int j = 0; j < ow; ++j)
			{
				nn_int start_h = i * stride_oh;
				nn_int start_w = j * stride_ow;
				for (nn_int c = 0; c < channels; ++c)
				{
					nn_float *prow_c = prow + fw * fh * c;
					nn_float *pimg = (nn_float*)img + iw * ih * c;
					for (nn_int v = 0; v < fh; ++v)
					{
						nn_int ir = start_h + v * stride_iw;
						for (nn_int u = 0; u < fw; ++u)
						{
							nn_int ic = start_w + u * stride_ih;
							prow_c[u + v * fw] = pimg[ic + ir * iw];
						}
					}
				}
				prow += fw * fh * channels;
			}
		}

	}

	static void conv_input_w(const varray &in_img, mem_block &block, const varray &filters, nn_int stride_w, nn_int stride_h, varray &out_img)
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
			img2row(&in_img(0, 0, 0), in_w, in_h, in_d, filter_w, filter_h, 1, 1, w, h, stride_w, stride_h, block);

			gemm(block.data, block.w, block.h
				, (nn_float*)&filters(0, 0, 0, k), 1, filter_w * filter_h * filter_d
				, &out_img(0, 0, k), 1, w * h);

		}
	}

	static void conv_input_delta(const varray &in_img, mem_block &block, const varray &delta, nn_int stride_w, nn_int stride_h, varray &dw)
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
				const nn_float *img_c = &in_img(0, 0, c);
				const nn_float *delta_k = &delta(0, 0, k);
				nn_float *dw_ck = &dw(0, 0, c, k);

				img2row(&in_img(0, 0, c), in_w, in_h, 1, delta_w, delta_h, stride_w, stride_h, w, h, 1, 1, block);

				gemm(block.data, block.w, block.h
					, (nn_float*)&delta(0, 0, k), 1, delta_w * delta_h
					, &dw(0, 0, c, k), 1, w * h);

			}
		}
	}

	static bool is_pad(nn_int a, nn_int stride, nn_int dr, nn_int size)
	{
		if (dr * stride != a)
		{
			return true;
		}
		if (dr < 0 || dr >= size)
		{
			return true;
		}
		return false;
	}

	static void conv_delta_w(const varray &delta, mem_block &block
		, const varray &filters, nn_int stride_w, nn_int stride_h, varray &ret)
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

		/*
			delta(u, v) = sum_i_j( delta(i, j) * w(u - stride_w * i, v - stride_h * j) )
		*/

		for (nn_int c = 0; c < d; ++c)
		{
			nn_float *ret_c = &ret(0, 0, c);
			for (nn_int k = 0; k < filter_count; ++k)
			{
				const nn_float *delta_k = &delta(0, 0, k);
				const nn_float *filter_c_k = &filters(0, 0, c, k);

				block.w = filter_w * filter_h;
				block.h = w * h;
				nn_float *prow = block.data;
				for (nn_int i = 0; i < h; ++i)
				{
					for (nn_int j = 0; j < w; ++j)
					{
						for (nn_int r = 0; r < filter_h; ++r)
						{
							for (nn_int c = 0; c < filter_w; ++c)
							{
								nn_int dc = (j - c) / stride_w;
								nn_int dr = (i - r) / stride_h;
								prow[c + r * filter_w] = is_pad(j - c, stride_w, dc, delta_w) || is_pad(i - r, stride_h, dr, delta_h)
									? 0 : delta_k[dc + dr * delta_w];
							}
						}
						prow += filter_w * filter_h;
					}
				}

				gemm(block.data, block.w, block.h
					, (nn_float*)&filters(0, 0, c, k), 1, filter_w * filter_h
					, ret_c, 1, w * h);

			}
		}
	}
#else //nnGEMM
static void conv_input_w(const varray &in_img, mem_block &block, const varray &filters, nn_int stride_w, nn_int stride_h, varray &out_img)
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
			conv_2d(&in_img(0, 0, c), in_w, in_h, block.data
				, 1, 1
				, &filters(0, 0, c, k), filter_w, filter_h
				, stride_w, stride_h
				, &out_img(0, 0, k), w, h);
		}
	}
}

static void conv_input_delta(const varray &in_img, mem_block &block, const varray &delta, nn_int stride_w, nn_int stride_h, varray &dw)
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
			conv_2d(&in_img(0, 0, c), in_w, in_h, block.data
				, stride_w, stride_h
				, &delta(0, 0, k), delta_w, delta_h
				, 1, 1
				, &dw(0, 0, c, k), w, h);
		}
	}
}

static void conv_delta_w(const varray &delta, mem_block &block, const varray &filters, nn_int stride_w, nn_int stride_h, varray &ret)
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
			conv_2d<true>(&delta(0, 0, k), delta_w, delta_h, block.data
				, stride_w, stride_h
				, &filters(0, 0, c, k), filter_w, filter_h
				, 1, 1
				, &ret(0, 0, c), w, h);
		}
	}
}

/*
common 2d convolution
*/
template<bool filter_flip = false>
static inline void conv_2d(const nn_float *img, nn_int iw, nn_int ih, nn_float *data
	, nn_int stride_iw, nn_int stride_ih
	, const nn_float *filter, nn_int fw, nn_int fh
	, nn_int stride_ow, nn_int stride_oh
	, nn_float *out, nn_int ow, nn_int oh)
{
	if (!filter_flip)
	{
		for (nn_int i = 0; i < oh; ++i)
		{
			for (nn_int j = 0; j < ow; ++j)
			{
				nn_int start_h = i * stride_oh;
				nn_int start_w = j * stride_ow;
				nn_float dot = 0;
				for (nn_int r = 0; r < fh; ++r)
				{
					nn_int v = start_h + r * stride_ih;
					for (nn_int c = 0; c < fw; ++c)
					{
						nn_int u = start_w + c * stride_iw;
						data[c + r * fw] = img[u + v * iw];
					}
				}
				out[j + i * ow] += vec_dot(data, filter, fw * fh);
			}
		}
	}
	else
	{
		for (nn_int i = 0; i < oh; ++i)
		{
			for (nn_int j = 0; j < ow; ++j)
			{
				nn_int idx = 0;
				for (nn_int r = 0; r < fh; ++r)
				{
					nn_int v = (i - r) / stride_ih;
					bool vpad = (v < 0 || v >= ih || v * stride_ih != i - r);
					for (nn_int c = 0; c < fw; ++c)
					{
						nn_int u = (j - c) / stride_iw;
						data[idx++] = (vpad || (u < 0 || u >= iw || u * stride_iw != j - c)) ? 0 : img[u + v * iw];
					}
				}
				out[j + i * ow] += vec_dot(data, filter, fw * fh);
			}
		}
	}

}

#endif //nnGEMM

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

