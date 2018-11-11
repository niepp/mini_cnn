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
		nn_int in_w = m_prev->m_out_shape.m_w;
		nn_int in_h = m_prev->m_out_shape.m_h;
		nn_int in_d = m_prev->m_out_shape.m_d;

		nn_int out_w = m_out_shape.m_w;
		nn_int out_h = m_out_shape.m_h;
		nn_int out_d = m_out_shape.m_d;

		m_task_storage.resize(task_count);
		for (auto& ts : m_task_storage)
		{
			if (!m_out_shape.is_img())
			{
				ts.m_x.resize(out_w * out_h * out_d);
			}
			else
			{
				ts.m_x.resize(out_w, out_h, out_d);
			}
			ts.m_wd.resize(in_w, in_h, in_d);
		}

		m_max_pooling_task_storage.resize(task_count);
		for (auto &pooling_ts : m_max_pooling_task_storage)
		{
			pooling_ts.m_idx_maps.resize(in_d);
			for (auto &mp : pooling_ts.m_idx_maps)
			{
				mp.resize(out_w * out_h);
			}
		}
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		varray &out_x = m_task_storage[task_idx].m_x;

		std::vector<index_vec> &idx_maps = m_max_pooling_task_storage[task_idx].m_idx_maps;

		down_sample(input, out_x, idx_maps, m_pool_w, m_pool_h, m_stride_w, m_stride_h);

		if (m_next != nullptr)
		{
			m_next->forw_prop(out_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, nn_int task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];
		std::vector<index_vec> &idx_maps = m_max_pooling_task_storage[task_idx].m_idx_maps;

		up_sample(next_wd, ts.m_wd, idx_maps, m_pool_w, m_pool_h, m_stride_w, m_stride_h);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	static void down_sample(const varray &in_img, varray &out, std::vector<index_vec> &idx_map,
		nn_int pool_w, nn_int pool_h,
		nn_int pool_stride_w, nn_int pool_stride_h)
	{

		nn_int in_w = in_img.width();
		nn_int in_h = in_img.height();
		nn_int in_d = in_img.depth();
		
		nn_int w = out.width();
		nn_int h = out.height();
		nn_int d = out.depth();

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
							nn_float t = in_img(x, y, c);
							if (t > maxv)
							{
								maxv = t;
								pool_idx = u + v * pool_w;
							}
						}
					}
					out(i, j, c) = maxv;
					if (pool_idx >= 0)
					{
						nn_int out_idx = i + j * w;
						idx_map[c][out_idx] = pool_idx;
					}
				}
			}
		}

	}

	static void up_sample(const varray &in_img, varray &out, const std::vector<index_vec> &idx_map,
		nn_int pool_w, nn_int pool_h,
		nn_int pool_stride_w, nn_int pool_stride_h)
	{
		nn_int in_w = in_img.width();
		nn_int in_h = in_img.height();
		nn_int in_d = in_img.depth();

		nn_int w = out.width();
		nn_int h = out.height();
		nn_int d = out.depth();

		nn_int map_d = static_cast<nn_int>(idx_map.size());

		nn_assert(map_d == in_d && map_d == d && map_d > 0);

		nn_int map_sz = static_cast<nn_int>(idx_map[0].size());

		nn_assert(map_sz == in_w * in_h);

		out.make_zero();

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
						out(x, y, c) += in_img(i, j, c);
					}
				}
			}
		}

	}
};
}
#endif //__MAX_POOLING_LAYER_H__

