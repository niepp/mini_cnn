#ifndef __MAX_POOLING_LAYER_H__
#define __MAX_POOLING_LAYER_H__

namespace mini_cnn 
{

class max_pooling_layer : public layer_base
{

protected:
	nn_int m_pool_w;
	nn_int m_pool_h;
	nn_int m_stride_w;
	nn_int m_stride_h;

	struct max_pooling_task_storage
	{
		/*
			https://software.intel.com/sites/products/documentation/doclib/daal/daal-user-and-reference-guides/daal_prog_guide/GUID-2C3AA967-AE6A-4162-84EB-93BE438E3A05.htm
			m_idx_maps[d]: index map of channel d
			m_idx_maps[d][i]: the i-th element of downsample output is choosed from the m_idx_maps[d][i] (index of pool window)
			for example
			input: 2X3		    pool: size 2X2,		    output: 1X2
									  stride 1X1						idx_map: 1X2
			| 0  2  1 |         forward																								 | 2  1 |
			| 3  5  4 |            =>					| 5  5 |        | 3  2 | : 5 in output[1] is the 2-th element of pool windos | 5  4 |
		*/ 
		std::vector<index_vec> m_idx_maps;
	};

	std::vector<max_pooling_task_storage> m_max_pooling_task_storage;

public:
	max_pooling_layer(nn_int pool_w, nn_int pool_h, nn_int stride_w, nn_int stride_h)
		: layer_base()
		, m_pool_w(pool_w), m_pool_h(pool_h)
		, m_stride_w(stride_w), m_stride_h(stride_h)
	{
		nn_assert(stride_w > 0 && stride_w <= pool_w);
		nn_assert(stride_h > 0 && stride_h <= pool_h);
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);

		nn_assert(m_prev->m_out_shape.is_img());

		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		nn_int out_w = static_cast<nn_int>(::floorf(1.0f * (in_w - m_pool_w) / m_stride_w)) + 1;
		nn_int out_h = static_cast<nn_int>(::floorf(1.0f * (in_h - m_pool_h) / m_stride_h)) + 1;
		m_out_shape.set(out_w, out_h, in_d);
	}

	virtual void set_task_count(nn_int task_count)
	{
		layer_base::set_task_count(task_count);
		m_task_storage.resize(task_count);
	}

	virtual void set_batch_size(nn_int batch_size)
	{
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;
		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int out_d = m_out_shape.m_d;
		if (!m_out_shape.is_img())
		{
			m_x_vec.resize(out_w * out_h * out_d, 1, 1, batch_size);
		}
		else
		{
			m_x_vec.resize(out_w, out_h, out_d, batch_size);
		}
		m_wd_vec.resize(in_w, in_h, in_d, batch_size);

		m_max_pooling_task_storage.resize(batch_size);
		for (auto &pooling_ts : m_max_pooling_task_storage)
		{
			pooling_ts.m_idx_maps.resize(in_d);
			for (auto &mp : pooling_ts.m_idx_maps)
			{
				mp.resize(out_w * out_h);
			}
		}

	}

	virtual void forw_prop(const varray &input_batch)
	{
		nn_int in_w = input_batch.width();
		nn_int in_h = input_batch.height();
		nn_int in_d = input_batch.depth();
		nn_int batch_size = input_batch.count();

		nn_int w = m_x_vec.width();
		nn_int h = m_x_vec.height();
		nn_int d = m_x_vec.depth();

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx) {
			for (int b = begin; b < end; ++b)
			{
				std::vector<index_vec> &idx_maps = m_max_pooling_task_storage[b].m_idx_maps;
				down_sample(input_batch.data(b), in_w, in_h, in_d
					, m_x_vec.data(b), w, h, d
					, idx_maps, m_pool_w, m_pool_h, m_stride_w, m_stride_h);
			}
		});

		if (m_next != nullptr)
		{
			m_next->forw_prop(m_x_vec);
		}
	}

	virtual void back_prop(const varray &next_wd)
	{
		nn_int in_w = next_wd.width();
		nn_int in_h = next_wd.height();
		nn_int in_d = next_wd.depth();
		nn_int batch_size = next_wd.count();

		nn_int w = m_wd_vec.width();
		nn_int h = m_wd_vec.height();
		nn_int d = m_wd_vec.depth();

		m_wd_vec.make_zero();

		parallel_task(batch_size, m_task_count, [&](nn_int begin, nn_int end, nn_int task_idx) {
			for (int b = begin; b < end; ++b)
			{
				std::vector<index_vec> &idx_maps = m_max_pooling_task_storage[b].m_idx_maps;
				up_sample(next_wd.data(b), in_w, in_h, in_d
					, m_wd_vec.data(b), w, h, d
					, idx_maps, m_pool_w, m_pool_h, m_stride_w, m_stride_h);
			}
		});

		m_prev->back_prop(m_wd_vec);

	}

private:
	static void down_sample(const nn_float *nn_restrict in_img, nn_int in_w, nn_int in_h, nn_int in_d
		, nn_float *nn_restrict out, nn_int w, nn_int h, nn_int d
		, std::vector<index_vec> &idx_map,
		nn_int pool_w, nn_int pool_h,
		nn_int pool_stride_w, nn_int pool_stride_h)
	{

		nn_int map_d = static_cast<nn_int>(idx_map.size());

		nn_assert(map_d == in_d && map_d == d && map_d > 0);

		nn_int map_sz = static_cast<nn_int>(idx_map[0].size());

		nn_assert(map_sz == w * h);

		for (auto &mp : idx_map)
		{
			for (nn_int i = 0; i < map_sz; ++i)
			{
				mp[i] = -1;
			}
		}

		for (nn_int c = 0; c < d; ++c)
		{
			for (nn_int i = 0; i < w; ++i)
			{
				for (nn_int j = 0; j < h; ++j)
				{
					nn_int start_w = i * pool_stride_w;
					nn_int start_h = j * pool_stride_h;
					nn_float maxv = cMinFloat;
					nn_int pool_idx = -1;
					for (nn_int v = 0; v < pool_h; ++v)
					{
						nn_int y = start_h + v;
						for (nn_int u = 0; u < pool_w; ++u)
						{
							nn_int x = start_w + u;
							nn_float t = in_img[x + y * in_w + c * in_w * in_h];
							if (t > maxv)
							{
								maxv = t;
								pool_idx = u + v * pool_w;
							}
						}
					}
					out[i + j * w + c * w * h] = maxv;
					if (pool_idx >= 0)
					{
						nn_int out_idx = i + j * w;
						idx_map[c][out_idx] = pool_idx;
					}
				}
			}
		}

	}

	static void up_sample(const nn_float *nn_restrict in_img, nn_int in_w, nn_int in_h, nn_int in_d		
		, nn_float *nn_restrict out, nn_int w, nn_int h, nn_int d
		, const std::vector<index_vec> &idx_map,
		nn_int pool_w, nn_int pool_h,
		nn_int pool_stride_w, nn_int pool_stride_h)
	{
		nn_int map_d = static_cast<nn_int>(idx_map.size());

		nn_assert(map_d == in_d && map_d == d && map_d > 0);

		nn_int map_sz = static_cast<nn_int>(idx_map[0].size());

		nn_assert(map_sz == in_w * in_h);

		for (nn_int c = 0; c < in_d; ++c)
		{
			for (nn_int i = 0; i < in_w; ++i)
			{
				for (nn_int j = 0; j < in_h; ++j)
				{
					nn_int start_w = i * pool_stride_w;
					nn_int start_h = j * pool_stride_h;

					nn_int out_idx = i + j * in_w;
					nn_int pool_idx = idx_map[c][out_idx];
					if (pool_idx >= 0)
					{
						nn_int v = pool_idx / pool_w;
						nn_int u = pool_idx - v * pool_w;
						nn_int x = start_w + u;
						nn_int y = start_h + v;
						nn_assert((x >= 0 && x < w) && (y >= 0 && y < h));
						out[x + y * w + c * w * h] += in_img[i + j * in_w + c * in_w * in_h];
					}
				}
			}
		}

	}
};
}
#endif //__MAX_POOLING_LAYER_H__

