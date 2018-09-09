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

const Float cAbsPrecision = 1e-4;
const Float cRelatePrecision = 1e-2;
const Int cinput_w = 12;
const Int cinput_h = 12;
const Int cinput_d = 1;
Int cinput_n = cinput_w * cinput_h * cinput_d;
Int coutput_n = 10;

Network create_fcn_sigmod_mse()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_n));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eMSE, eActiveFunc::eSigmod));
	return nn;
}

Network create_fcn_sigmod_crossentropy()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_n));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSigmod_CrossEntropy, eActiveFunc::eSigmod));
	return nn;
}

Network create_fcn_relu()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_n));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eRelu));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

Network create_fcn_softmax()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_n));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

Network create_cnn_sigmod()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_w, cinput_h, cinput_d));
	nn.AddLayer(new ConvolutionalLayer(4, new FilterDimension(3, 3, 1, 0, 1, 1), nullptr, eActiveFunc::eSigmod));
	nn.AddLayer(new ConvolutionalLayer(8, new FilterDimension(3, 3, 4, 0, 1, 1), nullptr, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eMSE, eActiveFunc::eSigmod));
	return nn;
}

Network create_cnn_sigmod_softmax()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_w, cinput_h, cinput_d));
	nn.AddLayer(new ConvolutionalLayer(4, new FilterDimension(3, 3, 1, 0, 1, 1), nullptr, eActiveFunc::eSigmod));
	nn.AddLayer(new ConvolutionalLayer(16, new FilterDimension(3, 3, 4, 0, 1, 1), nullptr, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

Network create_cnn_sigmod_softmax_pool()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_w, cinput_h, cinput_d));
	nn.AddLayer(new ConvolutionalLayer(4, new FilterDimension(3, 3, 1, 0, 1, 1), new Pooling(2, 2, 0, 1, 1), eActiveFunc::eSigmod));
//	nn.AddLayer(new ConvolutionalLayer(16, new FilterDimension(3, 3, 4, 0, 1, 1), new Pooling(2, 2, 0, 2, 2), eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

Network create_cnn_relu_mse()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_w, cinput_h, cinput_d));
	nn.AddLayer(new ConvolutionalLayer(4, new FilterDimension(3, 3, 1, 0, 1, 1), nullptr, eActiveFunc::eRelu));
	nn.AddLayer(new ConvolutionalLayer(5, new FilterDimension(3, 3, 4, 0, 1, 1), nullptr, eActiveFunc::eRelu));
	nn.AddLayer(new ConvolutionalLayer(6, new FilterDimension(3, 3, 5, 0, 1, 1), nullptr, eActiveFunc::eRelu));
	nn.AddLayer(new FullyConnectedLayer(16, eActiveFunc::eRelu));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eMSE, eActiveFunc::eRelu));
	return nn;
}

Network create_cnn_relu_softmax()
{
	Network nn;
	nn.AddLayer(new InputLayer(cinput_w, cinput_h, cinput_d));
	nn.AddLayer(new ConvolutionalLayer(4, new FilterDimension(3, 3, 1, 0, 1, 1), nullptr, eActiveFunc::eRelu));
	nn.AddLayer(new ConvolutionalLayer(5, new FilterDimension(3, 3, 4, 0, 1, 1), nullptr, eActiveFunc::eRelu));
	nn.AddLayer(new FullyConnectedLayer(12, eActiveFunc::eRelu));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

bool test_nn_gradient_check(Network &nn, const NormalRandom &nrand, VectorN *input, VectorN *label, Float absPrecision = 1e-4, Float relatePrecision = 1e-2)
{
	nn.Init(nrand);
	return nn.GradientCheck(*input, *label, absPrecision, relatePrecision);
}

#define TEST_GRADIENT(model)\
	cout << setw(30) << setiosflags(ios::left) << #model << "\t" << std::boolalpha << test_nn_gradient_check(model(), nRand, input, label, cAbsPrecision, cRelatePrecision) << endl;

void CheckGradient()
{
	uInt seed = GetNow();

	seed = 2572007265;
	std::mt19937_64 generator(seed);
	NormalRandom nRand(generator, 0, 1.0);
	UniformRandom uRand(generator, 0, 1.0);

	VectorN *input = new VectorN(cinput_n, 0);
	VectorN *label = new VectorN(coutput_n, 0);
	for (Int i = 0; i < input->GetSize(); ++i)
	{
		(*input)[i] = uRand.GetRandom();
	}
	(*label)[3] = 1.0;

	//TEST_GRADIENT(create_fcn_sigmod_mse);

	//TEST_GRADIENT(create_fcn_sigmod_crossentropy);

	//TEST_GRADIENT(create_fcn_relu);

	//TEST_GRADIENT(create_fcn_softmax);

	//TEST_GRADIENT(create_cnn_sigmod);

	//TEST_GRADIENT(create_cnn_sigmod_softmax);

	TEST_GRADIENT(create_cnn_sigmod_softmax_pool); // check failed with pooling

	//TEST_GRADIENT(create_cnn_relu_mse);

	//TEST_GRADIENT(create_cnn_relu_softmax);

}

int main()
{
	CheckGradient();
	system("pause");
	return 0;
}

