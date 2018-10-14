#ifndef __AVG_POOLING_LAYER_H__
#define __AVG_POOLING_LAYER_H__

namespace mini_cnn 
{

class avg_pooling_layer : public layer_base
{

protected:
	int_t m_pool_w;
	int_t m_pool_h;
	int_t m_stride_w;
	int_t m_stride_h;

public:
	avg_pooling_layer(int_t pool_w, int_t pool_h, int_t stride_w, int_t stride_h)
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

		int_t in_w = m_prev->m_out_shape.m_w;
		int_t in_h = m_prev->m_out_shape.m_h;
		int_t in_d = m_prev->m_out_shape.m_d;
		int_t out_w = static_cast<int_t>(::floorf(1.0f * (in_w - m_pool_w) / m_stride_w)) + 1;
		int_t out_h = static_cast<int_t>(::floorf(1.0f * (in_h - m_pool_h) / m_stride_h)) + 1;
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
	}

	virtual void forw_prop(const varray &input, int_t task_idx)
	{
		varray &out_x = m_task_storage[task_idx].m_x;

		down_sample(input, out_x, m_pool_w, m_pool_h, m_stride_w, m_stride_h);

		if (m_next != nullptr)
		{
			m_next->forw_prop(out_x, task_idx);
		}
	}

	virtual void back_prop(const varray &next_wd, int_t task_idx)
	{
		layer_base::task_storage &ts = m_task_storage[task_idx];
		const varray &input = m_prev->get_output(task_idx);

		int_t out_sz = next_wd.size();
		int_t in_sz = input.size();

		up_sample(next_wd, ts.m_wd, m_pool_w, m_pool_h, m_stride_w, m_stride_h);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	static void down_sample(const varray &in_img, varray &out, int_t pool_w, int_t pool_h, int_t pool_stride_w, int_t pool_stride_h)
	{

		int_t in_w = in_img.width();
		int_t in_h = in_img.height();
		int_t in_d = in_img.depth();

		int_t w = out.width();
		int_t h = out.height();
		int_t d = out.depth();

		nn_assert(in_d == d);

		float_t inv_Size = cOne / (pool_w * pool_h);

		for (int_t c = 0; c < d; ++c)
		{
			for (int_t i = 0; i < w; ++i)
			{
				for (int_t j = 0; j < h; ++j)
				{
					int_t start_w = i * pool_stride_w;
					int_t start_h = j * pool_stride_h;
					float_t s = 0;
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
							s += in_img(x, y, c);
						}
					}
					out(i, j, c) = s * inv_Size;
				}
			}
		}

	}

	static void up_sample(const varray &in_img, varray &out, int_t pool_w, int_t pool_h, int_t pool_stride_w, int_t pool_stride_h)
	{
		int_t in_w = in_img.width();
		int_t in_h = in_img.height();
		int_t in_d = in_img.depth();

		int_t w = out.width();
		int_t h = out.height();
		int_t d = out.depth();

		nn_assert(in_d == d);

		float_t inv_Size = cOne / (pool_w * pool_h);

		out.make_zero();

		for (int_t c = 0; c < in_d; ++c)
		{
			for (int_t i = 0; i < in_w; ++i)
			{
				for (int_t j = 0; j < in_h; ++j)
				{
					int_t start_w = i * pool_stride_w;
					int_t start_h = j * pool_stride_h;
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
							out(x, y, c) += in_img(i, j, c) * inv_Size;
						}
					}
				}
			}
		}

	}

};
}
#endif //__AVG_POOLING_LAYER_H__

