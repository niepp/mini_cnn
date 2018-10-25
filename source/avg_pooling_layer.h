#ifndef __AVG_POOLING_LAYER_H__
#define __AVG_POOLING_LAYER_H__

namespace mini_cnn 
{

class avg_pooling_layer : public layer_base
{

protected:
	nn_int m_pool_w;
	nn_int m_pool_h;
	nn_int m_stride_w;
	nn_int m_stride_h;

public:
	avg_pooling_layer(nn_int pool_w, nn_int pool_h, nn_int stride_w, nn_int stride_h)
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
	}

	virtual void forw_prop(const varray &input, nn_int task_idx)
	{
		varray &out_x = m_task_storage[task_idx].m_x;

		down_sample(input, out_x, m_pool_w, m_pool_h, m_stride_w, m_stride_h);

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

		up_sample(next_wd, ts.m_wd, m_pool_w, m_pool_h, m_stride_w, m_stride_h);

		m_prev->back_prop(ts.m_wd, task_idx);

	}

private:
	static void down_sample(const varray &in_img, varray &out, nn_int pool_w, nn_int pool_h, nn_int pool_stride_w, nn_int pool_stride_h)
	{

		nn_int in_w = in_img.width();
		nn_int in_h = in_img.height();
		nn_int in_d = in_img.depth();

		nn_int w = out.width();
		nn_int h = out.height();
		nn_int d = out.depth();

		nn_assert(in_d == d);

		nn_float inv_Size = cOne / (pool_w * pool_h);

		for (nn_int c = 0; c < d; ++c)
		{
			for (nn_int i = 0; i < w; ++i)
			{
				for (nn_int j = 0; j < h; ++j)
				{
					nn_int start_w = i * pool_stride_w;
					nn_int start_h = j * pool_stride_h;
					nn_float s = 0;
					for (nn_int u = 0; u < pool_w; ++u)
					{
						nn_int x = start_w + u;
						if (x < 0 || x >= in_w)
						{
							continue;
						}
						for (nn_int v = 0; v < pool_h; ++v)
						{
							nn_int y = start_h + v;
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

	static void up_sample(const varray &in_img, varray &out, nn_int pool_w, nn_int pool_h, nn_int pool_stride_w, nn_int pool_stride_h)
	{
		nn_int in_w = in_img.width();
		nn_int in_h = in_img.height();
		nn_int in_d = in_img.depth();

		nn_int w = out.width();
		nn_int h = out.height();
		nn_int d = out.depth();

		nn_assert(in_d == d);

		nn_float inv_Size = cOne / (pool_w * pool_h);

		out.make_zero();

		for (nn_int c = 0; c < in_d; ++c)
		{
			for (nn_int i = 0; i < in_w; ++i)
			{
				for (nn_int j = 0; j < in_h; ++j)
				{
					nn_int start_w = i * pool_stride_w;
					nn_int start_h = j * pool_stride_h;
					for (nn_int u = 0; u < pool_w; ++u)
					{
						nn_int x = start_w + u;
						if (x < 0 || x >= w)
						{
							continue;
						}
						for (nn_int v = 0; v < pool_h; ++v)
						{
							nn_int y = start_h + v;
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

