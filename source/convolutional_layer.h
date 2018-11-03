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
		varray m_padded_img;
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

		nn_int temp_w = in_w + in_w - out_w;
		nn_int temp_h = in_h + in_h - out_h;
		nn_int img2row_size = temp_w * temp_h * m_filter_shape.m_w * m_filter_shape.m_h;
		m_conv_task_storage.resize(task_count);
		for (auto &cts : m_conv_task_storage)
		{
			cts.m_padded_img.resize(temp_w, temp_h, out_d);
			cts.m_img_block.resize(img2row_size);
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		varray &out_z = m_task_storage[task_idx].m_z;
		varray &out_x = m_task_storage[task_idx].m_x;

		mem_block &block = m_conv_task_storage[task_idx].m_img_block;
		conv_input_w(input, block, m_w, m_stride_w, m_stride_h, out_z);

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
		conv_input_delta(input, block, ts.m_delta, m_stride_w, m_stride_h, ts.m_dw);

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

		//varray &padded_img = m_conv_task_storage[task_idx].m_padded_img;
		//padded_img.make_zero();

		//copy_to_temp(ts.m_delta, 0, 0
		//	, padded_img, offset_w, offset_h
		//	, out_w, out_h);

		conv_delta_w(ts.m_delta, block, m_w, m_stride_w, m_stride_h, ts.m_wd);

		//copy_to_temp(padded_img, offset_w, offset_h
		//	, ts.m_delta, 0, 0
		//	, out_w, out_h);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	static inline void copy_to_temp(const varray &src, nn_int offset_src_w, nn_int offset_src_h
		, varray &dst, nn_int offset_dst_w, nn_int offset_dst_h
		, nn_int w, nn_int h)
	{
		nn_int src_w = src.width();
		nn_int src_h = src.height();
		nn_int src_d = src.depth();

		nn_int dst_w = dst.width();
		nn_int dst_h = dst.height();
		nn_int dst_d = dst.depth();

		nn_assert(src_d == dst_d);
		nn_assert(w + offset_src_w <= src_w);
		nn_assert(h + offset_src_h <= src_h);
		nn_assert(w + offset_dst_w <= dst_w);
		nn_assert(h + offset_dst_h <= dst_h);

		for (nn_int c = 0; c < src_d; ++c)
		{
			for (nn_int i = 0; i < h; ++i)
			{
				nn_float *nn_restrict vec_dst = &dst(offset_dst_w, i + offset_dst_h, c);
				const nn_float *nn_restrict vec_src = &src(offset_src_w, i + offset_src_h, c);
				for (nn_int j = 0; j < w; ++j)
				{
					vec_dst[j] = vec_src[j];
				}
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
			for (nn_int c = 0; c < in_d; ++c)
			{
				conv_2d(&in_img(0, 0, c), in_w, in_h, block.data
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
				const nn_float *img_c = &in_img(0, 0, c);
				const nn_float *delta_k = &delta(0, 0, k);
				nn_float *dw_ck = &dw(0, 0, c, k);

				for (nn_int i = 0; i < w; ++i)
				{
					for (nn_int j = 0; j < h; ++j)
					{
						nn_int start_w = i;
						nn_int start_h = j;
						nn_int idx = 0;
						for (nn_int r = 0; r < delta_h; ++r)
						{
							nn_int ir = start_h + r * stride_h;
							for (nn_int c = 0; c < delta_w; ++c)
							{
								nn_int ic = start_w + c * stride_w;
								block.data[idx] = img_c[ic + ir * in_w];
								++idx;
							}
						}
						dw_ck[i + j * w] += vec_dot(block.data, delta_k, delta_w * delta_h);
					}
				}

			}
		}
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

		for (nn_int c = 0; c < d; ++c)
		{
			nn_float *ret_c = &ret(0, 0, c);
			for (nn_int k = 0; k < filter_count; ++k)
			{
				const nn_float *delta_k = &delta(0, 0, k);
				const nn_float *filter_c_k = &filters(0, 0, c, k);

				for (nn_int u = 0; u < w; ++u)
				{
					for (nn_int v = 0; v < h; ++v)
					{
						nn_int idx = 0;
						for (nn_int i = 0; i < delta_h; ++i)
						{
							nn_int ir = v - i * stride_h;
							for (nn_int j = 0; j < delta_w; ++j)
							{
								nn_int ic = u - j * stride_w;
								if ((ir < 0 || ir >= filter_h) || (ic < 0 || ic >= filter_w))
								{
									block.data[idx] = 0;
								}
								else{
									block.data[idx] = filter_c_k[ic + ir * filter_w];
								}
								++idx;
							}
						}
						ret_c[u + v * w] += vec_dot(block.data, delta_k, delta_w * delta_h);
					}
				}

			}
		}
	}

	/*
		common 2d convolution
	*/
	template<bool filter_flip = false>
	static inline void conv_2d(const nn_float *img, nn_int iw, nn_int ih, nn_float *cache
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
					nn_int idx = 0;
					for (nn_int r = 0; r < fh; ++r)
					{
						nn_int ir = start_h + r;
						for (nn_int c = 0; c < fw; ++c)
						{
							nn_int ic = start_w + c;
							cache[idx] = img[ic + ir * iw];
							++idx;
						}
					}
					out[i + j * ow] += vec_dot(cache, filter, fw * fh);
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
					nn_int idx = 0;
					for (nn_int r = 0; r < fh; ++r)
					{
						nn_int ir = start_h - r;
						for (nn_int c = 0; c < fw; ++c)
						{
							nn_int ic = start_w - c;
							cache[idx] = img[ic + ir * iw];
							++idx;
						}
					}
					out[i + j * ow] += vec_dot(cache, filter, fw * fh);
				}
			}
		}

	}

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

