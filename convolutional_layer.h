#ifndef __CONVOLUTIONAL_LAYER_H__
#define __CONVOLUTIONAL_LAYER_H__

namespace mini_cnn 
{

class filter_dim
{
public:
	int_t m_width;
	int_t m_height;
	int_t m_channels;
	int_t m_padding;
	int_t m_stride_w;
	int_t m_stride_h;
	filter_dim(int_t w, int_t h, int_t c, int_t padding, int_t stride_w, int_t stride_h) :
		m_width(w), m_height(h), m_channels(c), m_padding(padding), m_stride_w(stride_w), m_stride_h(stride_h)
	{
	}
	filter_dim(const filter_dim &filterDim) :
		m_width(filterDim.m_width), m_height(filterDim.m_height), m_channels(filterDim.m_channels)
		, m_padding(filterDim.m_padding), m_stride_w(filterDim.m_stride_w), m_stride_h(filterDim.m_stride_h)
	{
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
public:
	//std::vector<Matrix3D*> m_filters;
	//VectorN *m_bias;

	//VectorN *m_db;	// dJ/d(bias)
	//std::vector<Matrix3D*> m_dw;	// dJ/d(w)
	//Matrix3D *m_delta;		// dJ/d(z)

protected:
	//Matrix3D *m_middle;	// middle value

	//Matrix3D *m_input_img;
	//Matrix3D *m_pre_pool_img;
	//Matrix3D *m_output_img;
	//std::vector<IndexVector*> m_idx_maps;

	//Matrix3D *m_pre_unpool_delta;

protected:
	filter_dim m_filter_dim;
	active_func m_f;
	active_func m_df;
public:
	convolutional_layer(int_t out_size
		, filter_dim *ftdim
		, activation_type act)
		: layer_base(out_size)
		, m_filter_dim(*ftdim)
	{
	}

	virtual void set_task_count(int_t task_count)
	{
	}

	virtual const varray& forw_prop(const varray& input, int_t task_idx)
	{
		return m_next->forw_prop(input, task_idx);
	}

	virtual const varray& back_prop(const varray& next_wd, int_t task_idx)
	{
		return next_wd;
	}

};
}
#endif //__CONVOLUTIONAL_LAYER_H__

