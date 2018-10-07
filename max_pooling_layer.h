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

	index_vec m_idx_maps;

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
		m_idx_maps.resize(in_w, in_h);
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
			ts.m_x.resize(out_w, out_h, out_d);
			//ts.m_delta.resize(out_w, out_h, out_d);
			//ts.m_wd.resize(in_w, in_h, in_d);
		}
	}

	virtual void forw_prop(const varray &input, int_t task_idx)
	{
		varray &out_x = m_task_storage[task_idx].m_x;

		down_sample(input, out_x, m_idx_maps, m_pool_shape.m_w, m_pool_shape.m_h, m_stride_w, m_stride_h);

		if (m_next != nullptr)
		{
			if (!m_next->m_out_shape.is_img())
			{
				out_x.reshape(out_x.size());
			}
			m_next->forw_prop(out_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, int_t task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];
		const varray &input = m_prev->get_output(task_idx);

		int_t out_sz = next_wd.size();
		int_t in_sz = input.size();

	
		for (int_t i = 0; i < out_sz; ++i)
		{
			ts.m_delta[i] *= next_wd[i];
		}

		// up_sample();
		
		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	static void down_sample(const varray &img, varray & out, index_vec &idx_map,
		int_t pool_w, int_t pool_h,
		int_t pool_stride_w, int_t pool_stride_h)
	{

		int_t img_w = img.width();
		int_t img_h = img.height();
		int_t img_d = img.depth();

		int_t map_sz = static_cast<int_t>(idx_map.size());

		nn_assert(map_sz == img_w * img_h);

		for (int_t i = 0; i < map_sz; ++i)
		{
			idx_map[i] = -1;
		}

		int_t w = out.width();
		int_t h = out.height();
		int_t d = out.depth();
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
						if (x < 0 || x >= img_w)
						{
							continue;
						}
						for (int_t v = 0; v < pool_h; ++v)
						{
							int_t y = start_h + v;
							if (y < 0 || y >= img_h)
							{
								continue;
							}
							float_t s = img(x, y, c);
							if (s > maxv)
							{
								maxv = s;
								in_idx = x + y * img_w;
							}
						}
					}
					out(i, j, c) = maxv;
					if (in_idx >= 0)
					{
						int_t out_idx = i + j * w;
						idx_map[in_idx] = out_idx;
					}
				}
			}
		}

	}

	static void up_sample(const varray &img, varray &out, index_vec &idx_map,
		int_t pool_w, int_t pool_h,
		int_t pool_stride_w, int_t pool_stride_h)
	{

		int_t img_w = img.width();
		int_t img_h = img.height();
		int_t img_d = img.depth();

		int_t map_sz = static_cast<int_t>(idx_map.size());

		int_t w = out.width();
		int_t h = out.height();
		int_t d = out.depth();

		nn_assert(map_sz == w * h);

		for (int_t c = 0; c < d; ++c)
		{
			for (int_t i = 0; i < w; ++i)
			{
				for (int_t j = 0; j < h; ++j)
				{
					int_t in_idx = i + j * w;
					int_t out_idx = idx_map[in_idx];
					if (out_idx >= 0)
					{
						int_t y = out_idx / img_w;
						int_t x = out_idx - y * img_w;
						out(i, j, c) = img(x, y, c);
					}
					else
					{
						out(i, j, c) = cZero;
					}
				}
			}
		}

	}
};
}
#endif //__MAX_POOLING_LAYER_H__

