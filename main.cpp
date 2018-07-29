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
#include "reshape_layer.h"
#include "flatten_layer.h"
#include "convolutional_layer.h"
#include "network.h"

using namespace std;

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
	nn.AddLayer(new ConvolutionalLayer(1, 3, 3, 1, 0, 1, 1, eActiveFunc::eSigmod));
	nn.AddLayer(new ConvolutionalLayer(1, 5, 5, 1, 0, 1, 1, eActiveFunc::eSigmod));
	nn.AddLayer(new OutputLayer(C_classCount, eLossFunc::eSoftMax_LogLikelihood, eActiveFunc::eSoftMax));
	return nn;
}

// todo
// 1.整理成矢量/矩阵形式
// 2.抽象出layer
// 3.损失函数使用交叉熵，输出层使用softmax激活函数
// 4.加入卷积层

int main()
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

	std::vector<VectorN> img_vec(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		img_vec[k].SetSize(N_inputCount);
		for (int i = 0; i < N_inputCount; ++i)
		{
			float v = images[index + k * N_inputCount + i] * 1.0f / 255.0f;
			img_vec[k][i] = v;
		}
	}

	std::vector<VectorN> lab_vec(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		lab_vec[k].SetSize(C_classCount);
		int lab = labels[idx + k];
		lab_vec[k][lab] = 1.0f;
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

	std::vector<VectorN> test_img_vec(test_img_count);
	for (int k = 0; k < test_img_count; ++k)
	{
		test_img_vec[k].SetSize(N_inputCount);
		for (int i = 0; i < N_inputCount; ++i)
		{
			float v = test_images[test_idx + k * N_inputCount + i] * 1.0f / 255.0f;
			test_img_vec[k][i] = v;
		}
	}

	std::vector<int> test_lab_vec(test_lab_count);
	for (int k = 0; k < test_lab_count; ++k)
	{		
		test_lab_vec[k]= test_labels[lab_idx + k];
	}

	// random init
	NormalRandom nrand(0, 1.0f);

	// define neural network
	Network nn = CreateCNN();

	nn.Init(nrand);

	float learning_rate = 3.0f;
	int epoch = 20;
	int batch_size = 100;
	int batch = img_count / batch_size;
	std::vector<int> idx_vec(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		idx_vec[k] = k;
	}

	long long t0 = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()).time_since_epoch().count();

	uint32_t seed = (uint32_t)(std::chrono::system_clock::now().time_since_epoch().count());
	std::mt19937_64 generator(seed);

	float maxCorrectRate = 0;
	for (int c = 0; c < epoch; ++c)
	{
		std::shuffle(std::begin(idx_vec), std::end(idx_vec), generator);
		std::vector<VectorN*> batch_img_vec(batch_size);
		std::vector<VectorN*> batch_label_vec(batch_size);
		for (int i = 0; i <= batch; ++i)
		{
			for (int k = 0; k < batch_size; ++k)
			{
				int j = idx_vec[(i * batch_size + k) % img_count];
				batch_img_vec[k] = &img_vec[j];
				batch_label_vec[k] = &lab_vec[j];
			}
			if (i % 100 == 0)
			{
				cout << "batch: " << i << "/" << batch << endl;
			}
			nn.SGD(batch_img_vec, batch_label_vec, learning_rate);
		}

		uint32_t correct = nn.Test(test_img_vec, test_lab_vec);
		float correct_rate = (1.0f * correct / test_img_count);
		if (correct_rate > maxCorrectRate)
		{
			maxCorrectRate = correct_rate;			
		}
		cout << "epoch " << c << ": " << correct_rate << " (" << correct << " / " << test_img_count << ")" << endl;
	}

	cout << "maxCorrectRate: " << maxCorrectRate << endl;

	long long t1 = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()).time_since_epoch().count();

	float timeCost = (t1 - t0) * 0.001f;
	cout << "cost: " << timeCost << "(s)" << endl;

	system("pause");
	return 0;

}

