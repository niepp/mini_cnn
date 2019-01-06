#ifndef __BATCH_NORMALIZATION_LAYER_H__
#define __BATCH_NORMALIZATION_LAYER_H__

namespace mini_cnn 
{

class batch_normalization_layer : public layer_base
{
protected:

public:
	batch_normalization_layer()
		: layer_base()
	{
	}

	virtual void set_task_count(nn_int task_count)
	{

	}

};
}
#endif //__BATCH_NORMALIZATION_LAYER_H__

