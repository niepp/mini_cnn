#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

#include "../mini_cnn.h"

using namespace std;
using namespace mini_cnn;

Network create_fcn_sigmod_mse()
{
	Network nn;
	nn.AddLayer(new InputLayer(N_inputCount));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eMSE, eActiveFunc::eSigmod));
	return nn;
}

Network create_fcn_sigmod_crossentropy()
{
	Network nn;
	nn.AddLayer(new InputLayer(N_inputCount));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSigmod_CrossEntropy, eActiveFunc::eSigmod));
	return nn;
}

Network create_fcn_relu()
{
	Network nn;
	nn.AddLayer(new InputLayer(N_inputCount));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eRelu));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

Network create_fcn_softmax()
{
	Network nn;
	nn.AddLayer(new InputLayer(N_inputCount));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

Network create_cnn_sigmod()
{
	Network nn;
	nn.AddLayer(new InputLayer(W_input, H_input, D_input));
	nn.AddLayer(new ConvolutionalLayer(4, new FilterDimension(3, 3, 1, 0, 1, 1), nullptr, eActiveFunc::eSigmod));
	nn.AddLayer(new ConvolutionalLayer(8, new FilterDimension(3, 3, 4, 0, 1, 1), nullptr, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eMSE, eActiveFunc::eSigmod));
	return nn;
}

Network create_cnn_relu()
{
	Network nn;
	nn.AddLayer(new InputLayer(W_input, H_input, D_input));
	nn.AddLayer(new ConvolutionalLayer(4, new FilterDimension(3, 3, 1, 0, 1, 1), new Pooling(2, 2, 0, 2, 2), eActiveFunc::eRelu));
	nn.AddLayer(new ConvolutionalLayer(16, new FilterDimension(3, 3, 4, 0, 1, 1), new Pooling(2, 2, 0, 2, 2), eActiveFunc::eRelu));
	nn.AddLayer(new ConvolutionalLayer(120, new FilterDimension(3, 3, 16, 0, 1, 1), nullptr, eActiveFunc::eRelu));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

bool test_nn_gradient_check(Network &nn, const NormalRandom &nrand, VectorN *input, VectorN *label)
{
	nn.Init(nrand);
	return nn.GradientCheck(*input, *label);
}

void CheckGradient()
{
	uInt seed = GetNow();
	std::mt19937_64 generator(seed);
	NormalRandom nRand(generator, 0, 1.0);
	UniformRandom uRand(generator, 0, 1.0);

	VectorN *input = new VectorN(784, 0);
	VectorN *label = new VectorN(10, 0);
	for (Int i = 0; i < input->GetSize(); ++i)
	{
		(*input)[i] = uRand.GetRandom();
	}
	(*label)[3] = 1.0;

	bool check_ok = false;
	check_ok = test_nn_gradient_check(create_fcn_sigmod_mse(), nRand, input, label);
	cout << "create_fcn_sigmod_mse:\t" << std::boolalpha << check_ok << endl;

	check_ok = test_nn_gradient_check(create_fcn_sigmod_crossentropy(), nRand, input, label);
	cout << "create_fcn_sigmod_crossentropy:\t" << std::boolalpha << check_ok << endl;

	check_ok = test_nn_gradient_check(create_fcn_relu(), nRand, input, label);
	cout << "create_fcn_relu:\t" << std::boolalpha << check_ok << endl;

	check_ok = test_nn_gradient_check(create_fcn_softmax(), nRand, input, label);
	cout << "create_fcn_softmax:\t" << std::boolalpha << check_ok << endl;

	check_ok = test_nn_gradient_check(create_cnn_relu(), nRand, input, label);
	cout << "create_cnn_relu:\t" << std::boolalpha << check_ok << endl;

	check_ok = test_nn_gradient_check(create_cnn_sigmod(), nRand, input, label);
	cout << "create_cnn_sigmod:\t" << std::boolalpha << check_ok << endl;
}

int main()
{
	CheckGradient();
	system("pause");
	return 0;
}

