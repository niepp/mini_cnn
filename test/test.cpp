#include <iostream>
#include <iomanip>

#define GRADIENT_CHECKER
#include "../source/mini_cnn.h"

namespace mini_cnn
{

std::mt19937_64 global_setting::m_rand_generator = std::mt19937_64(2572007265);// fixed seed to repeat test
//std::mt19937_64 global_setting::m_rand_generator = std::mt19937_64(get_now_ms());

class gradient_checker
{
private:
	const nn_int cInput_w = 18;
	const nn_int cInput_h = 18;
	const nn_int cInput_d = 1;
	const nn_int cInput_n = cInput_w * cInput_h * cInput_d;
	const nn_int cOutput_n = 10;

public:
#define TEST_GRADIENT(model)\
	std::cout << std::setw(50) << std::setiosflags(std::ios::left) << #model << "\t" << std::boolalpha << test_nn_gradient_check(model(), input, label) << std::endl;

	gradient_checker()
	{
		uniform_random uRand(0, 1.0);

		varray *input = new varray(cInput_n);
		varray *label = new varray(cOutput_n);
		for (nn_int i = 0; i < input->size(); ++i)
		{
			(*input)[i] = uRand.get_random();
		}
		(*label)[3] = 1.0;

		TEST_GRADIENT(create_fcn_sigmod_mse);

		TEST_GRADIENT(create_fcn_sigmod_crossentropy);

		TEST_GRADIENT(create_fcn_softmax_loglikelihood);

		TEST_GRADIENT(create_fcn_relu);

		TEST_GRADIENT(create_fcn_relu_dropout);

		TEST_GRADIENT(create_fcn_softmax);

		TEST_GRADIENT(create_cnn_sigmod);

		TEST_GRADIENT(create_cnn_sigmod_dropout);

		TEST_GRADIENT(create_cnn_stride_2x2_sigmod);

		TEST_GRADIENT(create_cnn_padsame_stride_2x2_sigmod);

		TEST_GRADIENT(create_cnn_sigmod_softmax);

		TEST_GRADIENT(create_cnn_sigmod_softmax_max_pool);

		TEST_GRADIENT(create_cnn_sigmod_softmax_max_pool_overlap);

		TEST_GRADIENT(create_cnn_sigmod_softmax_avg_pool);

		TEST_GRADIENT(create_cnn_sigmod_softmax_avg_pool_overlap);

		TEST_GRADIENT(create_cnn_relu_mse);

		TEST_GRADIENT(create_cnn_relu_softmax);

		TEST_GRADIENT(create_cnn_activation_relu_softmax);

		TEST_GRADIENT(create_cnn_relu_padsame_softmax);

		TEST_GRADIENT(create_cnn_relu_softmax_max_pool);

		TEST_GRADIENT(create_cnn_relu_softmax_avg_pool);

		TEST_GRADIENT(create_cnn_relu_padsame_softmax_max_pool);

		TEST_GRADIENT(create_cnn_relu_padsame_softmax_avg_pool);

	}

private:
	bool test_nn_gradient_check(network &nn, varray *input, varray *label)
	{
		truncated_normal_initializer initializer(0, 0.1f, 2);
		nn.init_all_weight(initializer);
		return nn.gradient_check(*input, *label);
	}

	network create_fcn_sigmod_mse()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eSigmod));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eMSE, activation_type::eSigmod));
		return nn;
	}

	network create_fcn_sigmod_crossentropy()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eSigmod));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSigmod_CrossEntropy, activation_type::eSigmod));
		return nn;
	}

	network create_fcn_softmax_loglikelihood()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_fcn_relu()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(100, activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_fcn_relu_dropout()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(20, activation_type::eRelu));
		nn.add_layer(new dropout_layer((nn_float)(0.5)));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_fcn_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_sigmod()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 2, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(3, 3, 2, 3, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eMSE, activation_type::eSigmod));
		return nn;
	}

	network create_cnn_sigmod_dropout()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 2, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(3, 3, 2, 3, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new fully_connected_layer(30, activation_type::eSigmod));
		nn.add_layer(new dropout_layer((nn_float)(0.5)));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eMSE, activation_type::eSigmod));
		return nn;
	}

	network create_cnn_stride_2x2_sigmod()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 2, 2, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(3, 3, 4, 8, 2, 2, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eMSE, activation_type::eSigmod));
		return nn;
	}

	network create_cnn_padsame_stride_2x2_sigmod()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 2, 2, padding_type::eSame, activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(3, 3, 4, 8, 2, 2, padding_type::eSame, activation_type::eSigmod));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eMSE, activation_type::eSigmod));
		return nn;
	}

	network create_cnn_sigmod_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(3, 3, 4, 8, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_sigmod_softmax_max_pool()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new convolutional_layer(3, 3, 4, 16, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_sigmod_softmax_max_pool_overlap()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new max_pooling_layer(2, 2, 1, 1));
		nn.add_layer(new convolutional_layer(3, 3, 4, 16, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new max_pooling_layer(2, 2, 1, 1));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_sigmod_softmax_avg_pool()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new avg_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new convolutional_layer(3, 3, 4, 16, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new avg_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_sigmod_softmax_avg_pool_overlap()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new avg_pooling_layer(2, 2, 1, 1));
		nn.add_layer(new convolutional_layer(3, 3, 4, 16, 1, 1, padding_type::eValid, activation_type::eSigmod));
		nn.add_layer(new avg_pooling_layer(2, 2, 1, 1));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_relu_mse()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new convolutional_layer(3, 3, 5, 6, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(16, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eMSE, activation_type::eRelu));
		return nn;
	}

	network create_cnn_relu_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(12, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_activation_relu_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid));
		nn.add_layer(new activation_layer(activation_type::eRelu));
		nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eValid));
		nn.add_layer(new activation_layer(activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(12));
		nn.add_layer(new activation_layer(activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_relu_padsame_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eSame, activation_type::eRelu));
	//	nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eSame, activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(12, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_relu_softmax_max_pool()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new fully_connected_layer(12, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_relu_softmax_avg_pool()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new avg_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eValid, activation_type::eRelu));
		nn.add_layer(new avg_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new fully_connected_layer(12, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_relu_padsame_softmax_max_pool()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eSame, activation_type::eRelu));
		nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eSame, activation_type::eRelu));
		nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new fully_connected_layer(12, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_relu_padsame_softmax_avg_pool()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w, cInput_h, cInput_d));
		nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eSame, activation_type::eRelu));
		nn.add_layer(new avg_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new convolutional_layer(3, 3, 4, 5, 1, 1, padding_type::eSame, activation_type::eRelu));
		nn.add_layer(new avg_pooling_layer(2, 2, 2, 2));
		nn.add_layer(new fully_connected_layer(12, activation_type::eRelu));
		nn.add_layer(new output_layer(cOutput_n, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

};

}

int main()
{
	mini_cnn::gradient_checker();
	system("pause");
	return 0;
}

