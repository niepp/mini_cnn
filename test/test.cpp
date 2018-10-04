#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <cassert>
#include <random>
#include <algorithm>

#include "../mini_cnn.h"

namespace mini_cnn
{

class gradient_checker
{
	const float_t cPrecision = 1e-4;
	const int_t cInput_w = 12;
	const int_t cInput_h = 12;
	const int_t cInput_d = 1;
	const int_t cInput_n = cInput_w * cInput_h * cInput_d;
	const int_t cOutput_n = 10;
public:
#define TEST_GRADIENT(model)\
std::cout << std::setw(30) << std::setiosflags(std::ios::left) << #model << "\t" << std::boolalpha << test_nn_gradient_check(model(), nRand, input, label, cPrecision) << std::endl;

	gradient_checker()
	{
		int_t seed = get_now();
		//seed = 2572007265;
		std::mt19937_64 generator(seed);
		normal_random nRand(generator, 0, 1.0);
		uniform_random uRand(generator, 0, 1.0);

		varray *input = new varray(cInput_n);
		varray *label = new varray(cOutput_n);
		for (int_t i = 0; i < input->size(); ++i)
		{
			(*input)[i] = uRand.get_random();
		}
		(*label)[3] = 1.0;

		TEST_GRADIENT(create_fcn_sigmod_mse);

		TEST_GRADIENT(create_fcn_sigmod_crossentropy);

		TEST_GRADIENT(create_fcn_relu);

		TEST_GRADIENT(create_fcn_softmax);

		TEST_GRADIENT(create_cnn_sigmod);

		TEST_GRADIENT(create_cnn_sigmod_softmax);

		TEST_GRADIENT(create_cnn_sigmod_softmax_pool);

		TEST_GRADIENT(create_cnn_relu_mse);

		TEST_GRADIENT(create_cnn_relu_softmax);

	}

private:
	network create_fcn_sigmod_mse()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eSigmod));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eMSE, activation_type::eSigmod));
		return nn;
	}

	network create_fcn_sigmod_crossentropy()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eSigmod));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSigmod_CrossEntropy, activation_type::eSigmod));
		return nn;
	}

	network create_fcn_relu()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_fcn_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_n));
		nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_sigmod()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w * cInput_h * cInput_d));
		nn.add_layer(new convolutional_layer(4, new filter_dim(3, 3, 1, 0, 1, 1), activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(8, new filter_dim(3, 3, 4, 0, 1, 1), activation_type::eSigmod));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eMSE, activation_type::eSigmod));
		return nn;
	}

	network create_cnn_sigmod_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w * cInput_h * cInput_d));
		nn.add_layer(new convolutional_layer(4, new filter_dim(3, 3, 1, 0, 1, 1), activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(16, new filter_dim(3, 3, 4, 0, 1, 1), activation_type::eSigmod));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_sigmod_softmax_pool()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w * cInput_h * cInput_d));
		nn.add_layer(new convolutional_layer(4, new filter_dim(3, 3, 1, 0, 1, 1), activation_type::eSigmod));
		nn.add_layer(new convolutional_layer(16, new filter_dim(3, 3, 4, 0, 1, 1), activation_type::eSigmod));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	network create_cnn_relu_mse()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w * cInput_h * cInput_d));
		nn.add_layer(new convolutional_layer(4, new filter_dim(3, 3, 1, 0, 1, 1), activation_type::eRelu));
		nn.add_layer(new convolutional_layer(5, new filter_dim(3, 3, 4, 0, 1, 1), activation_type::eRelu));
		nn.add_layer(new convolutional_layer(6, new filter_dim(3, 3, 5, 0, 1, 1), activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(16, activation_type::eRelu));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eMSE, activation_type::eRelu));
		return nn;
	}

	network create_cnn_relu_softmax()
	{
		network nn;
		nn.add_layer(new input_layer(cInput_w * cInput_h * cInput_d));
		nn.add_layer(new convolutional_layer(4, new filter_dim(3, 3, 1, 0, 1, 1), activation_type::eRelu));
		nn.add_layer(new convolutional_layer(5, new filter_dim(3, 3, 4, 0, 1, 1), activation_type::eRelu));
		nn.add_layer(new fully_connected_layer(12, activation_type::eRelu));
		nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
		return nn;
	}

	bool test_nn_gradient_check(network &nn, const normal_random &nrand, varray *input, varray *label, float_t precision = 1e-4)
	{
		nn.init_all_weight(nrand);
		return nn.gradient_check(*input, *label, precision);
	}
};

}

int main()
{
	gradient_checker();
	system("pause");
	return 0;
}

