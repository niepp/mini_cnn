#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

#define nnGEMM

namespace mini_cnn 
{

class mem_block
{
	nn_int m_w;
	nn_int m_h;
	nn_float *m_data;
	nn_int m_data_len;
public:
	mem_block() : m_data(nullptr), m_data_len(0), m_w(0), m_h(0)
	{
	}

	void create(nn_int len)
	{
		m_data = (nn_float*)align_malloc(len * sizeof(nn_float), nn_align_size);
		memset(m_data, 0, len * sizeof(nn_float));
		m_data_len = len;
		m_w = 0;
		m_h = 0;
	}

	void set_size(nn_int w, nn_int h)
	{
		nn_assert(m_w * m_h <= m_data_len);
		m_w = w;
		m_h = h;
	}

	~mem_block() {
		release();
	}

	nn_float* data() {
		return m_data;
	}

	nn_int width() const {
		return m_w;
	}

	nn_int height() const {
		return m_h;
	}

private:
	void release()
	{
		if (m_data != nullptr) {
			align_free(m_data);
		}
		m_data = nullptr;
		m_data_len = 0;
		m_w = 0;
		m_h = 0;
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

	struct conv_task_storage
	{
		mem_block m_block_img;
		varray m_filter_cache;
	};
	std::vector<conv_task_storage> m_conv_task_storage;
	std::vector<nn_int> m_index_map;

public:
	convolutional_layer(nn_int filter_w, nn_int filter_h, nn_int filter_c, nn_int filter_n, nn_int stride_w, nn_int stride_h
		, padding_type padding, activation_type ac_type = activation_type::eIdentity) : layer_base(ac_type)
		, m_filter_shape(filter_w, filter_h, filter_c)
		, m_filter_count(filter_n), m_stride_w(stride_w), m_stride_h(stride_h), m_padding(padding)
	{
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);

		nn_assert(m_prev->m_out_shape.is_img());

		nn_assert(m_prev->m_out_shape.m_d == m_filter_shape.m_d);

		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;

		nn_int out_w = 0;
		nn_int out_h = 0;
		if (m_padding == padding_type::eValid)
		{
			out_w = static_cast<nn_int>(::floorf(1.0f * (in_w - m_filter_shape.m_w) / m_stride_w)) + 1;
			out_h = static_cast<nn_int>(::floorf(1.0f * (in_h - m_filter_shape.m_h) / m_stride_h)) + 1;
		}
		else
		{
			// todo add padding size around img
			out_w = in_w;
			out_h = in_h;
		}
		m_out_shape.set(out_w, out_h, m_filter_count);

		nn_int fw = m_filter_shape.m_w;
		nn_int fh = m_filter_shape.m_h;
		nn_int fd = m_filter_shape.m_d;

		m_b.resize(m_filter_count);
		m_w.resize(fw, fh, fd, m_filter_count);

		m_index_map.resize(fw * fh * in_w * in_h);
		bake_index_map(m_index_map, out_w, out_h
			, m_stride_w, m_stride_h
			, fw, fh
			, in_w, in_h);

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

		nn_int fw = m_filter_shape.m_w;
		nn_int fh = m_filter_shape.m_h;
		nn_int fd = m_filter_shape.m_d;

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
		nn_int im2col_size1 = (fw * fh * fd) * (out_w * out_h);
		nn_int im2col_size2 = (in_w * in_h * in_d) * (fw * fh);
		nn_int im2col_size3 = (fw * fh * m_filter_count) * (in_w * in_h);
		nn_int block_size = std::max(im2col_size1, im2col_size2);
		block_size = std::max(block_size, im2col_size3);
#else
		nn_int block_size = std::max(out_w * out_h, m_filter_shape.size());
#endif
		m_conv_task_storage.resize(task_count);
		for (auto &cts : m_conv_task_storage)
		{
			cts.m_block_img.create(block_size);
			cts.m_filter_cache.resize(fw * fh * fd * m_filter_count);
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		varray &out_z = m_task_storage[task_idx].m_z;
		varray &out_x = m_task_storage[task_idx].m_x;

		mem_block &block = m_conv_task_storage[task_idx].m_block_img;
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
		mem_block &block = m_conv_task_storage[task_idx].m_block_img;
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

#ifdef nnGEMM
		varray &filter_cache = m_conv_task_storage[task_idx].m_filter_cache;
		conv_delta_w(ts.m_delta, block, filter_cache, m_index_map, m_w, m_stride_w, m_stride_h, ts.m_wd);
#else
		conv_delta_w(ts.m_delta, block, m_index_map, m_w, m_stride_w, m_stride_h, ts.m_wd);
#endif
		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	static void bake_index_map(std::vector<nn_int> &index_map, nn_int iw, nn_int ih
		, nn_int stride_iw, nn_int stride_ih
		, nn_int fw, nn_int fh
		, nn_int ow, nn_int oh)
	{
		for (nn_int i = 0; i < oh; ++i)
		{
			for (nn_int j = 0; j < ow; ++j)
			{
				nn_int *pmap = &index_map[(j + i * ow) * fw * fh];
				for (nn_int r = 0; r < fh; ++r)
				{
					nn_int v = (i - r) / stride_ih;
					bool vpad = (v < 0 || v >= ih || v * stride_ih != i - r);
					for (nn_int c = 0; c < fw; ++c)
					{
						nn_int u = (j - c) / stride_iw;
						pmap[c + r * fw] = (vpad || (u < 0 || u >= iw || u * stride_iw != j - c)) ? -1 : u + v * iw;
					}
				}
			}
		}
	}

#ifdef nnGEMM
	static inline void im2col(const nn_float *img, nn_int iw, nn_int ih, nn_int channels
		, nn_int fw, nn_int fh
		, nn_int stride_iw, nn_int stride_ih
		, nn_int ow, nn_int oh
		, nn_int stride_ow, nn_int stride_oh
		, nn_float *prow, nn_int row_width)
	{
		for (nn_int i = 0; i < oh; ++i)
		{
			for (nn_int j = 0; j < ow; ++j)
			{
				nn_int start_h = i * stride_oh;
				nn_int start_w = j * stride_ow;
				nn_int idx = 0;
				for (nn_int c = 0; c < channels; ++c)
				{
					nn_float *pimg = (nn_float*)img + iw * ih * c;
					for (nn_int v = 0; v < fh; ++v)
					{
						nn_int ir = start_h + v * stride_ih;
						for (nn_int u = 0; u < fw; ++u)
						{
							nn_int ic = start_w + u * stride_iw;
							prow[idx++] = (ir >= ih || ic >= iw) ? 0 : pimg[ic + ir * iw];
						}
					}
				}
				prow += row_width;
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

		block.set_size(filter_w * filter_h * filter_d, w * h);
		im2col(&in_img(0, 0, 0), in_w, in_h, in_d, filter_w, filter_h, 1, 1, w, h, stride_w, stride_h, block.data(), block.width());

		for (nn_int k = 0; k < filter_count; ++k)
		{
			fo_mv_v(block.data(), block.height(), block.width()
				, &filters(0, 0, 0, k)
				, &out_img(0, 0, k));
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

		block.set_size(delta_w * delta_h, w * h * d);
		for (nn_int c = 0; c < d; ++c)
		{
			nn_float *prow = block.data() + delta_w * delta_h * w * h * c;
			im2col(&in_img(0, 0, c), in_w, in_h, 1, delta_w, delta_h, stride_w, stride_h, w, h, 1, 1, prow, block.width());
		}

		for (nn_int k = 0; k < n; ++k)
		{
			fo_mv_v(block.data(), block.height(), block.width()
				, &delta(0, 0, k)
				, &dw(0, 0, 0, k));
		}

	}

	static void conv_delta_w(const varray &delta, mem_block &block, varray &filter_cache, std::vector<nn_int> &index_map
		, const varray &filters, nn_int stride_w, nn_int stride_h, varray &wd)
	{
		nn_int delta_w = delta.width();
		nn_int delta_h = delta.height();
		nn_int delta_d = delta.depth();

		nn_int filter_count = filters.count();
		nn_int filter_w = filters.width();
		nn_int filter_h = filters.height();
		nn_int filter_d = filters.depth();

		nn_int in_w = wd.width();
		nn_int in_h = wd.height();
		nn_int in_d = wd.depth();

		nn_assert(delta.check_dim(3));
		nn_assert(filters.check_dim(4));
		nn_assert(wd.check_dim(3));

		nn_assert(delta_d == filter_count);
		nn_assert(in_d == filter_d);

		/*
			delta(u, v) = sum_i_j( delta(i, j) * w(u - stride_w * i, v - stride_h * j) )
		*/

		nn_int filter_size = filter_w * filter_h;
		block.set_size(filter_size * filter_count, in_w * in_h);
		nn_float *prow = block.data();

		for (nn_int i = 0; i < in_h; ++i)
		{
			for (nn_int j = 0; j < in_w; ++j)
			{
				nn_int *pmap = &index_map[(j + i * in_w) * filter_size];
				for (nn_int k = 0; k < filter_count; ++k)
				{
					const nn_float *delta_k = &delta(0, 0, k);
					for (nn_int idx = 0; idx < filter_size; ++idx)
					{
						prow[idx] = pmap[idx] >= 0 ? delta_k[pmap[idx]] : 0;
					}
					prow += filter_size;
				}
			}
		}

		nn_float *nn_restrict pfilter = &filter_cache[0];
		for (nn_int c = 0; c < in_d; ++c)
		{
			for (nn_int k = 0; k < filter_count; ++k)
			{
				nn_float *nn_restrict filter_cache_k = pfilter + k * filter_size;
				const nn_float *nn_restrict filter_c_k = &filters(0, 0, c, k);
				for (nn_int idx = 0; idx < filter_size; ++idx)
				{
					filter_cache_k[idx] = filter_c_k[idx];
				}
			}
			pfilter += filter_count * filter_size;
		}

		gemm(&filter_cache[0], filter_count * filter_size, in_d
			, block.data(), block.width(), block.height()
			, &wd(0, 0, 0), in_w * in_h, in_d);

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

	for (nn_int k = 0; k < filter_count; ++k)
	{
		conv_3d(&in_img(0, 0, 0), in_w, in_h, in_d, block.data()
			, stride_w, stride_h
			, &filters(0, 0, 0, k), filter_w, filter_h
			, &out_img(0, 0, k), w, h);
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
			conv_2d(&in_img(0, 0, c), in_w, in_h, block.data()
				, stride_w, stride_h
				, &delta(0, 0, k), delta_w, delta_h
				, &dw(0, 0, c, k), w, h);
		}
	}

}

static void conv_delta_w(const varray &delta, mem_block &block, std::vector<nn_int> &index_map, const varray &filters, nn_int stride_w, nn_int stride_h, varray &wd)
{
	nn_int delta_w = delta.width();
	nn_int delta_h = delta.height();
	nn_int delta_d = delta.depth();

	nn_int filter_count = filters.count();
	nn_int filter_w = filters.width();
	nn_int filter_h = filters.height();
	nn_int filter_d = filters.depth();

	nn_int in_w = wd.width();
	nn_int in_h = wd.height();
	nn_int in_d = wd.depth();

	nn_assert(delta.check_dim(3));
	nn_assert(filters.check_dim(4));
	nn_assert(ret.check_dim(3));

	nn_assert(delta_d == filter_count);

	nn_assert(in_d == filter_d);

	wd.make_zero();

	for (nn_int c = 0; c < in_d; ++c)
	{
		for (nn_int k = 0; k < filter_count; ++k)
		{
			conv_2d_flip(&delta(0, 0, k), delta_w, delta_h, block.data()
				, index_map
				, &filters(0, 0, c, k), filter_w, filter_h
				, &wd(0, 0, c), in_w, in_h);
		}
	}

}

static inline void conv_3d(const nn_float *img, nn_int iw, nn_int ih, nn_int id, nn_float *data
	, nn_int stride_w, nn_int stride_h
	, const nn_float *filter, nn_int fw, nn_int fh
	, nn_float *out, nn_int ow, nn_int oh)
{
	nn_int fitler_sz = fw * fh * id;
	for (nn_int i = 0; i < oh; ++i)
	{
		for (nn_int j = 0; j < ow; ++j)
		{
			nn_int start_h = i * stride_h;
			nn_int start_w = j * stride_w;
			for (nn_int k = 0; k < id; ++k)
			{
				const nn_float *img_k = img + k * iw * ih;
				nn_int cidx = k * fw * fh;
				for (nn_int y = 0; y < fh; ++y)
				{
					nn_int v = start_h + y;
					for (nn_int x = 0; x < fw; ++x)
					{
						nn_int u = start_w + x;
						data[cidx + x + y * fw] = (u >= iw || v >= ih) ? 0 : img_k[u + v * iw];
					}
				}
			}
			out[j + i * ow] = vec_dot(data, filter, fitler_sz);
		}
	}
}

static inline void conv_2d(const nn_float *img, nn_int iw, nn_int ih, nn_float *data
	, nn_int stride_w, nn_int stride_h
	, const nn_float *filter, nn_int fw, nn_int fh
	, nn_float *out, nn_int ow, nn_int oh)
{
	for (nn_int i = 0; i < oh; ++i)
	{
		for (nn_int j = 0; j < ow; ++j)
		{
			nn_int start_h = i;
			nn_int start_w = j;
			for (nn_int r = 0; r < fh; ++r)
			{
				nn_int v = start_h + r * stride_h;
				for (nn_int c = 0; c < fw; ++c)
				{
					nn_int u = start_w + c * stride_w;
					data[c + r * fw] = (u >= iw || v >= ih) ? 0 : img[u + v * iw];
				}
			}
			out[j + i * ow] += vec_dot(data, filter, fw * fh);
		}
	}
}

static inline void conv_2d_flip(const nn_float *img, nn_int iw, nn_int ih, nn_float *data
	, std::vector<nn_int> &index_map
	, const nn_float *filter, nn_int fw, nn_int fh
	, nn_float *out, nn_int ow, nn_int oh)
{
	for (nn_int i = 0; i < oh; ++i)
	{
		for (nn_int j = 0; j < ow; ++j)
		{
			nn_int *pmap = &index_map[(j + i * ow) * fw * fh];
			for (nn_int idx = 0; idx < fw * fh; ++idx)
			{
				data[idx] = pmap[idx] >= 0 ? img[pmap[idx]] : 0;
			}
			out[j + i * ow] += vec_dot(data, filter, fw * fh);
		}
	}
}

#endif //nnGEMM

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

