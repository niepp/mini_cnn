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
	nn_int m_pad_w;
	nn_int m_pad_h;

	struct conv_task_storage
	{
		mem_block m_block_img;
		varray m_filter_cache;
	};
	std::vector<conv_task_storage> m_conv_task_storage;
	std::vector<nn_int> m_index_map;

public:
	convolutional_layer(nn_int filter_w, nn_int filter_h, nn_int filter_c, nn_int filter_n, nn_int stride_w, nn_int stride_h
		, nn_int pad_w, nn_int pad_h, activation_base *activation) : layer_base(activation)
		, m_filter_shape(filter_w, filter_h, filter_c)
		, m_filter_count(filter_n), m_stride_w(stride_w), m_stride_h(stride_h)
		, m_pad_w(pad_w), m_pad_h(pad_h)
	{
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);

		nn_assert(m_prev->m_out_shape.is_img());

		nn_assert(m_prev->m_out_shape.m_d == m_filter_shape.m_d);

		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;

		nn_int out_w = static_cast<nn_int>(::floorf(1.0f * (in_w + 2 * m_pad_w - m_filter_shape.m_w) / m_stride_w)) + 1;
		nn_int out_h = static_cast<nn_int>(::floorf(1.0f * (in_h + 2 * m_pad_h - m_filter_shape.m_h) / m_stride_h)) + 1;
		m_out_shape.set(out_w, out_h, m_filter_count);

		nn_int fw = m_filter_shape.m_w;
		nn_int fh = m_filter_shape.m_h;
		nn_int fd = m_filter_shape.m_d;

		m_b.resize(m_filter_count);
		m_w.resize(fw, fh, fd, m_filter_count);

		m_index_map.resize(fw * fh * (in_w + 2 * m_pad_w) * (in_h + 2 * m_pad_h));
		//m_index_map.resize((fw * fh) * (in_w * in_h));
		bake_index_map(m_index_map, out_w, out_h
			, m_stride_w, m_stride_h
			, fw, fh
			, in_w, in_h
			, m_pad_w, m_pad_h);

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
		layer_base::set_task_count(task_count);
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
			ts.m_delta.resize(out_w, out_h, out_d);
		}

		//nn_int paded_in_w = in_w + 2 * m_pad_w;
		//nn_int paded_in_h = in_h + 2 * m_pad_h;
		nn_int im2col_size1 = (fw * fh * fd) * (out_w * out_h);
		nn_int im2col_size2 = (in_w * in_h * in_d) * (fw * fh);
		nn_int im2col_size3 = (fw * fh * m_filter_count) * (in_w * in_h);
		nn_int block_size = std::max<nn_int>(im2col_size1, im2col_size2);
		block_size = std::max<nn_int>(block_size, im2col_size3);

		m_conv_task_storage.resize(task_count);
		for (auto &cts : m_conv_task_storage)
		{
			cts.m_block_img.create(block_size);
			cts.m_filter_cache.resize(fw * fh * fd * m_filter_count);
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

		m_z_vec.resize(out_w, out_h, out_d, batch_size);
		if (!m_out_shape.is_img())
		{
			m_x_vec.resize(out_w * out_h * out_d, batch_size);
		}
		else
		{
			m_x_vec.resize(out_w, out_h, out_d, batch_size);
		}
		m_wd_vec.resize(in_w, in_h, in_d, batch_size);
	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_int in_w = input_batch.width();
		nn_int in_h = input_batch.height();
		nn_int in_d = input_batch.depth();
		nn_int batch_size = input_batch.count();

		nn_int out_w = m_z_vec.width();
		nn_int out_h = m_z_vec.height();
		nn_int out_d = m_z_vec.depth();

		nn_int out_size = m_out_shape.size();

		nn_assert(input_batch.check_dim(4));
		nn_assert(m_z_vec.check_dim(4));

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx)
		{
			layer_base::task_storage &ts = m_task_storage[task_idx];
			mem_block &block = m_conv_task_storage[task_idx].m_block_img;

			for (int b = begin; b < end; ++b)
			{
				conv_input_w(input_batch.data(b), in_w, in_h, in_d
					, m_pad_w, m_pad_h
					, block, m_w, m_stride_w, m_stride_h
					, m_z_vec.data(b), out_w, out_h, out_d);

				for (nn_int k = 0; k < m_out_shape.m_d; ++k)
				{
					nn_float bk = m_b(k);
					for (nn_int i = 0; i < m_out_shape.m_h; ++i)
					{
						nn_float *nn_restrict vec_out_z = &m_z_vec(0, i, k, b);
						for (nn_int j = 0; j < m_out_shape.m_w; ++j)
						{
							vec_out_z[j] += bk;
						}
					}
				}

				nn_float *out_z = m_z_vec.data(b);
				nn_float *out_x = m_x_vec.data(b);
				m_activation->f(out_z, out_x, out_size);
			}
		});

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}
	}

	virtual void back_prop(const varray &next_wd)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;

		nn_int out_sz = next_wd.img_size();
		nn_assert(out_sz == m_out_shape.size());
		nn_assert(m_wd_vec.check_dim(4));
		nn_assert(in_w == m_wd_vec.width());
		nn_assert(in_h == m_wd_vec.height());
		nn_assert(in_d == m_wd_vec.depth());

		m_wd_vec.make_zero();

		nn_int batch_size = next_wd.count();

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx)
		{
			layer_base::task_storage &ts = m_task_storage[task_idx];
			const varray &input_batch = m_prev->get_output();

			nn_int delta_w = ts.m_delta.width();
			nn_int delta_h = ts.m_delta.height();
			nn_int delta_d = ts.m_delta.depth();

			for (int b = begin; b < end; ++b)
			{
				const nn_float *vec_input = input_batch.data(b);
				nn_float *nn_restrict vec_delta = &ts.m_delta[0];
				const nn_float *vec_next_wd = next_wd.data(b);
				/*
					delta := next_wd กั df(z)
				*/
				m_activation->df(m_z_vec.data(b), vec_delta, out_sz);

				for (nn_int i = 0; i < out_sz; ++i)
				{
					vec_delta[i] *= vec_next_wd[i];
				}

				/*
					dw_k := conv2d(input_d, delta_k)
				*/
				mem_block &block = m_conv_task_storage[task_idx].m_block_img;
				conv_input_delta(vec_input, in_w, in_h, in_d, m_pad_w, m_pad_h, block, vec_delta, delta_w, delta_h, delta_d, m_stride_w, m_stride_h, ts.m_dw);

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
				nn_float *vec_wd = m_wd_vec.data(b);
				varray &filter_cache = m_conv_task_storage[task_idx].m_filter_cache;
				conv_delta_w(ts.m_delta, block, filter_cache, m_index_map, m_w, m_stride_w, m_stride_h, vec_wd, in_w, in_h, in_d, m_pad_w, m_pad_h);
			}
		});

		m_prev->back_prop(m_wd_vec);

	}

private:
	static void bake_index_map(std::vector<nn_int> &index_map, nn_int ow, nn_int oh
		, nn_int stride_iw, nn_int stride_ih
		, nn_int fw, nn_int fh
		, nn_int iw, nn_int ih
		, nn_int pad_w, nn_int pad_h)
	{
		for (nn_int i = 0; i < ih; ++i)
		{
			for (nn_int j = 0; j < iw; ++j)
			{
				nn_int *pmap = &index_map[(j + i * iw) * fw * fh];
				for (nn_int r = 0; r < fh; ++r)
				{
					nn_int v = (i - r + pad_h) / stride_ih;
					bool v_bad = (v < 0 || v >= oh || v * stride_ih != i - r + pad_h);
					for (nn_int c = 0; c < fw; ++c)
					{
						nn_int u = (j - c + pad_w) / stride_iw;
						bool u_bad = (u < 0 || u >= ow || u * stride_iw != j - c + pad_w);
						pmap[c + r * fw] = (v_bad || u_bad) ? -1 : u + v * ow;
					}
				}
			}
		}
	}

	static inline void im2col(const nn_float *img, nn_int iw, nn_int ih, nn_int channels
		, nn_int pad_w, nn_int pad_h
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
				nn_int start_h = i * stride_oh - pad_h;
				nn_int start_w = j * stride_ow - pad_w;
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
							prow[idx++] = (ir >= ih || ir < 0 || ic >= iw || ic < 0) ? 0 : pimg[ic + ir * iw];
						}
					}
				}
				prow += row_width;
			}
		}

	}

	static void conv_input_w(const nn_float *nn_restrict in_img, nn_int in_w, nn_int in_h, nn_int in_d
		, nn_int pad_w, nn_int pad_h
		, mem_block &block, const varray &filters, nn_int stride_w, nn_int stride_h
		, nn_float *nn_restrict out_img, nn_int out_w, nn_int out_h, nn_int out_d)
	{
		nn_int filter_count = filters.count();
		nn_int filter_w = filters.width();
		nn_int filter_h = filters.height();
		nn_int filter_d = filters.depth();

		nn_assert(filters.check_dim(4));

		nn_assert(in_d == filter_d);
		nn_assert(out_d == filter_count);

		block.set_size(filter_w * filter_h * filter_d, out_w * out_h);

		im2col(in_img, in_w, in_h, in_d, pad_w, pad_h, filter_w, filter_h, 1, 1, out_w, out_h, stride_w, stride_h, block.data(), block.width());

		nn_int bh = block.height();
		nn_int bw = block.width();
		gemm((nn_float)1.0
			, &filters(0, 0, 0, 0), filter_count, bw
			, block.data(), bh, bw
			, (nn_float)0.0
			, out_img, filter_count, bh);

	}

	static void conv_input_delta(const nn_float *nn_restrict in_img, nn_int in_w, nn_int in_h, nn_int in_d
		, nn_int pad_w, nn_int pad_h
		, mem_block &block
		, const nn_float *nn_restrict delta, nn_int delta_w, nn_int delta_h, nn_int delta_d
		, nn_int stride_w, nn_int stride_h, varray &dw)
	{
		nn_int w = dw.width();
		nn_int h = dw.height();
		nn_int d = dw.depth();
		nn_int n = dw.count();

		nn_assert(dw.check_dim(4));
		nn_assert(in_d == d);
		nn_assert(n == delta_d);

		block.set_size(delta_w * delta_h, w * h * d);
		for (nn_int c = 0; c < d; ++c)
		{
			nn_float *prow = block.data() + delta_w * delta_h * w * h * c;
			im2col(in_img + c * in_w * in_h, in_w, in_h, 1
				, pad_w, pad_h
				, delta_w, delta_h
				, stride_w, stride_h
				, w, h
				, 1, 1
				, prow, block.width());
		}

		//for (nn_int k = 0; k < n; ++k)
		//{
		//	fo_mv_v(block.data(), block.height(), block.width()
		//		, &delta(0, 0, k)
		//		, &dw(0, 0, 0, k));
		//}

		const nn_float *nn_restrict bptr = block.data();
		nn_int bh = block.height();
		nn_int bw = block.width();
		gemm((nn_float)1.0
			, delta, n, bw
			, bptr, bh, bw
			, (nn_float)1.0
			, &dw(0, 0, 0, 0), n, bh);

	}

	static void conv_delta_w(const varray &delta, mem_block &block, varray &filter_cache, std::vector<nn_int> &index_map
		, const varray &filters, nn_int stride_w, nn_int stride_h
		, nn_float *nn_restrict vec_wd, nn_int in_w, nn_int in_h, nn_int in_d
		, nn_int pad_w, nn_int pad_h)
	{
		nn_int delta_w = delta.width();
		nn_int delta_h = delta.height();
		nn_int delta_d = delta.depth();

		nn_int filter_count = filters.count();
		nn_int filter_w = filters.width();
		nn_int filter_h = filters.height();
		nn_int filter_d = filters.depth();

		nn_assert(delta.check_dim(3));
		nn_assert(filters.check_dim(4));

		nn_assert(delta_d == filter_count);
		nn_assert(in_d == filter_d);

		/*
			wd(u, v) = sum_i_j( delta(i, j) * w(u - stride_w * i, v - stride_h * j) )
		*/

		nn_int filter_size = filter_w * filter_h;
		//block.set_size(filter_size * filter_count, (in_w + 2 * pad_w) * (in_h + 2 * pad_h));
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

		gemm((nn_float)1.0
			, &filter_cache[0], in_d, filter_count * filter_size
			, block.data(), block.height(), block.width()
			, (nn_float)0.0
			, vec_wd, in_d, in_w * in_h);

	}

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

