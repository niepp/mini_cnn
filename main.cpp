#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <cassert>
//#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

#include "types.h"
#include "utils.h"
#include "math/vectorn.h"
#include "math/matrixmxn.h"
#include "math/matrix3d.h"
#include "math/mathdef.h"
#include "layer.h"
#include "input_layer.h"
#include "fully_connected_layer.h"
#include "output_layer.h"
#include "convolutional_layer.h"
#include "network.h"

using namespace std;
using namespace mini_cnn;

const int N_inputCount = 784;
const int W_input = 28;
const int H_input = 28;
const int D_input = 1;
const int C_classCount = 10;

int ReadInt(unsigned char *buffer, int &index)
{
	int vint = (buffer[index] << 24) | (buffer[index + 1] << 16) |
				(buffer[index + 2] << 8) | (buffer[index + 3]);
	index += 4;
	return vint;
}

unsigned char* ReadFile(const char *filePath)
{
	fstream fsread(filePath, std::fstream::in | std::fstream::binary);
	if (!fsread)
	{
		std::cerr << "Open failed!" << filePath << std::endl;
		return nullptr;
	}

	unsigned char *buffer;
	long size;

	//get length of file:
	fsread.seekg(0, std::ios::end);
	size = (long)fsread.tellg();
	fsread.seekg(0, std::ios::beg);

	buffer = new unsigned char[size];
	fsread.read(reinterpret_cast<char*>(buffer), size);
	fsread.close();

	return buffer;
}

Network CreateFCN()
{
	Network nn;
	nn.AddLayer(new InputLayer(N_inputCount));
	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

Network CreateCNN()
{
	Network nn;
	nn.AddLayer(new InputLayer(W_input, H_input, D_input));
	nn.AddLayer(new ConvolutionalLayer(6, new FilterDimension(3, 3, 1, 0, 1, 1), new Pooling(2, 2, 0, 2, 2), eActiveFunc::eRelu));
	nn.AddLayer(new ConvolutionalLayer(16, new FilterDimension(3, 3, 6, 0, 1, 1), new Pooling(2, 2, 0, 2, 2), eActiveFunc::eRelu));
	nn.AddLayer(new ConvolutionalLayer(120, new FilterDimension(3, 3, 16, 0, 1, 1), nullptr, eActiveFunc::eRelu));
//	nn.AddLayer(new FullyConnectedLayer(30, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eMSE, eActiveFunc::eSigmod));
	return nn;
}

void ReadDataSet(std::vector<VectorN*> &img_vec, std::vector<VectorN*> &lab_vec
	, std::vector<VectorN*> &test_img_vec, std::vector<int> &test_lab_vec)
{
	// read train data
	// train images
	unsigned char *images = ReadFile("data/train-images.idx3-ubyte");
	int index = 0;
	int img_migic = ReadInt(images, index);
	int img_count = ReadInt(images, index);
	int col = ReadInt(images, index);
	int row = ReadInt(images, index);

	// train labels
	unsigned char *labels = ReadFile("data/train-labels.idx1-ubyte");
	int idx = 0;
	int lab_migic = ReadInt(labels, idx);
	int lab_count = ReadInt(labels, idx);

	assert(img_count == lab_count);

	img_vec.resize(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		img_vec[k] = new VectorN(N_inputCount);
		for (int i = 0; i < N_inputCount; ++i)
		{
			float v = images[index + k * N_inputCount + i] * 1.0f / 255.0f;
			(*img_vec[k])[i] = v;
		}
	}

	lab_vec.resize(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		lab_vec[k] = new VectorN(C_classCount);
		int lab = labels[idx + k];
		(*lab_vec[k])[lab] = 1.0f;
	}

	// read test data
	// test images
	unsigned char *test_images = ReadFile("data/t10k-images.idx3-ubyte");
	int test_idx = 0;
	int test_img_migic = ReadInt(test_images, test_idx);
	int test_img_count = ReadInt(test_images, test_idx);
	col = ReadInt(test_images, test_idx);
	row = ReadInt(test_images, test_idx);

	// test labels
	unsigned char *test_labels = ReadFile("data/t10k-labels.idx1-ubyte");
	int lab_idx = 0;
	int test_lab_migic = ReadInt(test_labels, lab_idx);
	int test_lab_count = ReadInt(test_labels, lab_idx);

	assert(test_img_count == test_lab_count);

	test_img_vec.resize(test_img_count);
	for (int k = 0; k < test_img_count; ++k)
	{
		test_img_vec[k] = new VectorN(N_inputCount);
		for (int i = 0; i < N_inputCount; ++i)
		{
			float v = test_images[test_idx + k * N_inputCount + i] * 1.0f / 255.0f;
			(*test_img_vec[k])[i] = v;
		}
	}

	test_lab_vec.resize(test_lab_count);
	for (int k = 0; k < test_lab_count; ++k)
	{
		test_lab_vec[k] = test_labels[lab_idx + k];
	}
}

long long GetNow()
{
	return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()).time_since_epoch().count();
}

std::mt19937_64 CreateRandom()
{
	// random init
	uInt seed = (uInt)(std::chrono::system_clock::now().time_since_epoch().count());

	cout << "random seed:" << seed << endl;

	std::mt19937_64 generator(seed);

	return generator;
}

void CheckGradient()
{
	uInt seed = 566226029;
	std::mt19937_64 generator(seed);

	NormalRandom nrand(generator, 0, 1.0f);

	// define neural network
	Network nn = CreateFCN();

	nn.Init(nrand);

	std::vector<VectorN*> img_vec;
	std::vector<VectorN*> lab_vec;
	std::vector<VectorN*> test_img_vec;
	std::vector<int> test_lab_vec;
	ReadDataSet(img_vec, lab_vec, test_img_vec, test_lab_vec);

	VectorN *input = img_vec[0];
	VectorN *label = lab_vec[0];

	bool check_ok = nn.GradientCheck(*input, *label);
	cout << "GradientCheck: " << std::boolalpha << check_ok << endl;

}

int main()
{

	CheckGradient();

	system("pause");
	return 0;

	std::vector<VectorN*> img_vec;
	std::vector<VectorN*> lab_vec;
	std::vector<VectorN*> test_img_vec;
	std::vector<int> test_lab_vec;
	ReadDataSet(img_vec, lab_vec, test_img_vec, test_lab_vec);

	int img_count = img_vec.size();
	int test_img_count = test_img_vec.size();

	long long t0 = GetNow();
	auto generator = CreateRandom();
	NormalRandom nrand(generator, 0, 1.0f);

	// define neural network
	Network nn = CreateCNN();

	nn.Init(nrand);

	float learning_rate = 0.02f;
	int epoch = 30;
	int batch_size = 10;
	int batch = img_count / batch_size;
	std::vector<int> idx_vec(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		idx_vec[k] = k;
	}

	float rate = learning_rate;
	float maxCorrectRate = 0;
	for (int c = 0; c < epoch; ++c)
	{
		learning_rate = rate;
		std::shuffle(std::begin(idx_vec), std::end(idx_vec), generator);
		std::vector<VectorN*> batch_img_vec(batch_size);
		std::vector<VectorN*> batch_label_vec(batch_size);
		for (int i = 0; i <= batch; ++i)
		{
			for (int k = 0; k < batch_size; ++k)
			{
				int j = idx_vec[(i * batch_size + k) % img_count];
				batch_img_vec[k] = img_vec[j];
				batch_label_vec[k] = lab_vec[j];
			}
			if (i % (batch/5) == 0)
			{
				cout << "batch: " << i << "/" << batch << endl;
			}

			learning_rate *= 0.85f;
			learning_rate = std::max(0.00001f, learning_rate);

		//	Float cb = nn.CalcCost(batch_img_vec, batch_label_vec);
			nn.SGD(batch_img_vec, batch_label_vec, learning_rate);
		//	Float ca = nn.CalcCost(batch_img_vec, batch_label_vec);
		//	cout << "cost: " << cb << " -> " << ca << "\t" << cb - ca << endl;
		}

		uInt correct = nn.Test(test_img_vec, test_lab_vec);
		float correct_rate = (1.0f * correct / test_img_count);
		if (correct_rate > maxCorrectRate)
		{
			maxCorrectRate = correct_rate;
		}

		Float tot_cost = nn.CalcCost(img_vec, lab_vec);

		cout << "epoch " << c << ": " << correct_rate << " (" << correct << " / " << test_img_count << ")" << "\t tot_cost = " << tot_cost << endl;
	}

	cout << "Max CorrectRate: " << maxCorrectRate << endl;

	long long t1 = GetNow();

	float timeCost = (t1 - t0) * 0.001f;
	cout << "TimeCost: " << timeCost << "(s)" << endl;

	system("pause");
	return 0;

}

