#ifndef __MAX_POOLING_LAYER_H__
#define __MAX_POOLING_LAYER_H__

namespace mini_cnn 
{

class max_pooling_layer : public layer_base
{

protected:
	shape3d m_pool_shape;
	int_t m_stride_w;
	int_t m_stride_h;

	struct max_pooling_task_storage
	{
		std::vector<index_vec> m_idx_maps;
	};

	std::vector<max_pooling_task_storage> m_max_pooling_task_storage;

public:
	max_pooling_layer(int_t pool_w, int_t pool_h, int_t stride_w, int_t stride_h)
		: layer_base()
		, m_pool_shape(pool_w, pool_h, 1)
		, m_stride_w(stride_w), m_stride_h(stride_h)
	{
	}

	virtual void connect(layer_base *next)
	{
		layer_base::connect(next);

		nn_assert(m_prev->m_out_shape.is_img());

		int_t in_w = m_prev->m_out_shape.m_w;
		int_t in_h = m_prev->m_out_shape.m_h;
		int_t in_d = m_prev->m_out_shape.m_d;
		int_t out_w = static_cast<int_t>(::floorf(1.0f * (in_w - m_pool_shape.m_w) / m_stride_w)) + 1;
		int_t out_h = static_cast<int_t>(::floorf(1.0f * (in_h - m_pool_shape.m_h) / m_stride_h)) + 1;
		m_out_shape.set(out_w, out_h, in_d);
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
				mp.resize(in_w * in_h);
			}
		}
	}

	virtual void forw_prop(const varray &input, int_t task_idx)
	{
		varray &out_x = m_task_storage[task_idx].m_x;

		std::vector<index_vec> &idx_maps = m_max_pooling_task_storage[task_idx].m_idx_maps;

		down_sample(input, out_x, idx_maps, m_pool_shape.m_w, m_pool_shape.m_h, m_stride_w, m_stride_h);

		if (m_next != nullptr)
		{
			m_next->forw_prop(out_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, int_t task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];
		std::vector<index_vec> &idx_maps = m_max_pooling_task_storage[task_idx].m_idx_maps;

		up_sample(next_wd, ts.m_wd, idx_maps, m_pool_shape.m_w, m_pool_shape.m_h, m_stride_w, m_stride_h);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:

	// https://software.intel.com/sites/products/documentation/doclib/daal/daal-user-and-reference-guides/daal_prog_guide/GUID-2C3AA967-AE6A-4162-84EB-93BE438E3A05.htm
	static void down_sample(const varray &in_img, varray &out, std::vector<index_vec> &idx_map,
		int_t pool_w, int_t pool_h,
		int_t pool_stride_w, int_t pool_stride_h)
	{

		int_t in_w = in_img.width();
		int_t in_h = in_img.height();
		int_t in_d = in_img.depth();
		
		int_t w = out.width();
		int_t h = out.height();
		int_t d = out.depth();

		int_t map_d = static_cast<int_t>(idx_map.size());

		nn_assert(map_d == in_d && map_d == d && map_d > 0);

		int_t map_sz = static_cast<int_t>(idx_map[0].size());

		nn_assert(map_sz == in_w * in_h);

		for (auto &mp : idx_map)
		{
			for (int_t i = 0; i < map_sz; ++i)
			{
				mp[i] = -1;
			}
		}

		for (int_t c = 0; c < d; ++c)
		{
			for (int_t i = 0; i < w; ++i)
			{
				for (int_t j = 0; j < h; ++j)
				{
					int_t start_w = i * pool_stride_w;
					int_t start_h = j * pool_stride_h;
					float_t maxv = cMIN_FLOAT;
					int_t in_idx = -1;
					for (int_t u = 0; u < pool_w; ++u)
					{
						int_t x = start_w + u;
						if (x < 0 || x >= in_w)
						{
							continue;
						}
						for (int_t v = 0; v < pool_h; ++v)
						{
							int_t y = start_h + v;
							if (y < 0 || y >= in_h)
							{
								continue;
							}
							float_t t = in_img(x, y, c);
							if (t > maxv)
							{
								maxv = t;
								in_idx = x + y * in_w;
							}
						}
					}
					out(i, j, c) = maxv;
					if (in_idx >= 0)
					{
						int_t out_idx = i + j * w;
						idx_map[c][in_idx] = out_idx;
					}
				}
			}
		}

	}

	static void up_sample(const varray &in_img, varray &out, const std::vector<index_vec> &idx_map,
		int_t pool_w, int_t pool_h,
		int_t pool_stride_w, int_t pool_stride_h)
	{
		int_t in_w = in_img.width();
		int_t in_h = in_img.height();
		int_t in_d = in_img.depth();

		int_t w = out.width();
		int_t h = out.height();
		int_t d = out.depth();

		int_t map_d = static_cast<int_t>(idx_map.size());

		nn_assert(map_d == in_d && map_d == d && map_d > 0);

		int_t map_sz = static_cast<int_t>(idx_map[0].size());

		nn_assert(map_sz == w * h);

		out.make_zero();

		for (int_t c = 0; c < in_d; ++c)
		{
			for (int_t i = 0; i < in_w; ++i)
			{
				for (int_t j = 0; j < in_h; ++j)
				{
					int_t start_w = i * pool_stride_w;
					int_t start_h = j * pool_stride_h;
					float_t maxv = cMIN_FLOAT;
					int_t in_idx = -1;
					for (int_t u = 0; u < pool_w; ++u)
					{
						int_t x = start_w + u;
						if (x < 0 || x >= w)
						{
							continue;
						}
						for (int_t v = 0; v < pool_h; ++v)
						{
							int_t y = start_h + v;
							if (y < 0 || y >= h)
							{
								continue;
							}
							
							int_t idx = x + y * w;
							if (idx_map[c][idx] >= 0)
							{
								out(x, y, c) += in_img(i, j, c);
							}
						/*	else
							{
								out(x, y, c) = cZero;
							}*/
						}
					}
				}
			}
		}

	}
};
}
#endif //__MAX_POOLING_LAYER_H__

