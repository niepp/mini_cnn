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
#include "math/mathdef.h"

using namespace std;

const int N = 784;
const int M = 15;
const int C = 10;

// N -> M -> C
// z = w * a + b
// a = f(z)

// input layer
VectorN x(N);		// input(from image data)

// middle layer	
MatrixMN w(M, N);  // weight
VectorN b(M);		// bias
VectorN z(M);		// middle value
VectorN a(M);		// output of this layer

VectorN delta(M);	// equal to dJ/d(bias)
MatrixMN dw(M, N);

// output layer

MatrixMN wo(C, M);		// weight
VectorN bo(C);		// bias
VectorN zo(C);		// middle value
VectorN ao(C);		// output of this layer

VectorN deltao(C);    // equal to dJ/d(bias)
MatrixMN dwo(C, M);

VectorN y(C);		// label value form image data(one hot code)

// init weight and bias of all layers
// 使用均值0，方差1的归一化高斯分布
void init(NormalRandom nrand)
{
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			w(i, j) = nrand.GetRandom();
		}
	}

	for (int i = 0; i < M; ++i)
	{
		b[i] = nrand.GetRandom();
	}

	//
	for (int i = 0; i < C; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			wo(i, j) = nrand.GetRandom();
		}
	}

	for (int i = 0; i < C; ++i)
	{
		bo[i] = nrand.GetRandom();
	}

}

VectorN active_func(VectorN z)
{
	// sigmoid
	VectorN v(z.GetSize());
	for (unsigned int i = 0; i < z.GetSize(); ++i)
	{
		v[i] = 1.0f / (1.0f + exp(-z[i]));
	}
	return v;
}

VectorN active_func_prime(VectorN z)
{
	VectorN v(z.GetSize());
	for (unsigned int i = 0; i < z.GetSize(); ++i)
	{
		auto f = 1.0f / (1.0f + exp(-z[i]));
		v[i] = f * (1.0f - f);
	}
	return v;
}

// 损失函数对输出层的输出值的偏导数 （这里用均方误差函数MSE为例）
VectorN lost_derivate_aL()
{
	// nn output is ao[C]
	// label value form image is y[C]
	return ao - y;
}

void forward()
{
	// nn input is x[N]
	// z = w * a + b
	// a' = f(z)

	// middle layer
	z = (w * x) + b;
	a = active_func(z);

	// output layer
	zo = (wo * a) + bo;
	ao = active_func(zo);

}

void backprop()
{
	// output layer
	deltao = lost_derivate_aL() ^ active_func_prime(zo);

	// middle layer
	delta = (wo.Transpose() * deltao) ^ active_func_prime(z);

	// gradient of weight and bias
	dwo = deltao * a;

	dw = delta * x;

}

void train(const std::vector<VectorN> &batch_img_vec, const std::vector<VectorN> &batch_label_vec, float eta)
{

	MatrixMN edwo(dwo.GetRowCount(), dwo.GetColCount());
	edwo.MakeZero();
	VectorN edbo(deltao.GetSize());
	edbo.MakeZero();

	MatrixMN edw(dw.GetRowCount(), dw.GetColCount());
	edw.MakeZero();
	VectorN edb(delta.GetSize());
	edb.MakeZero();

	assert(batch_img_vec.size() == batch_label_vec.size());

	int batch_size = batch_img_vec.size();
	for (int i = 0; i < batch_size; ++i)
	{
		x = batch_img_vec[i];
		y = batch_label_vec[i];

		forward();
		backprop();

		edwo = edwo + dwo;
		edbo = edbo + deltao;

		edw = edw + dw;
		edb = edb + delta;
	}

	float r = eta / batch_size;

	// output layer
	wo = wo - edwo * r;
	bo = bo - edbo * r;

	// middle layer
	w = w - edw * r;
	b = b - edb * r;

}

uint32_t test(const std::vector<VectorN> &test_img_vec, const std::vector<int> &test_lab_vec)
{
	
	assert(test_img_vec.size() == test_lab_vec.size());

	int test_count = test_img_vec.size();
	uint32_t correct = 0;
	for (int k = 0; k < test_count; ++k)
	{
		x = test_img_vec[k];
		forward();
		auto maxlab = -1.0;
		int lab = -1;
		for (int i = 0; i < C; ++i)
		{
			if (ao[i] > maxlab)
			{
				maxlab = ao[i];
				lab = i;
			}
		}
		int std_lab = test_lab_vec[k];
		if (lab == std_lab)
		{
			++correct;
		}
	}

	return correct;

}

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
	std::cout << "length:" << size << std::endl;

	buffer = new unsigned char[size];
	fsread.read(reinterpret_cast<char*>(buffer), size);
	fsread.close();

	return buffer;
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
		img_vec[k].SetSize(N);
		for (int i = 0; i < N; ++i)
		{
			float v = images[index + k * N + i] * 1.0f / 255.0f;
			img_vec[k][i] = v;
		}
	}

	std::vector<VectorN> lab_vec(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		lab_vec[k].SetSize(C);
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
		test_img_vec[k].SetSize(N);
		for (int i = 0; i < N; ++i)
		{
			float v = test_images[test_idx + k * N + i] * 1.0f / 255.0f;
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

	init(nrand);

	float learning_rate = 1.25f;
	int epoch = 30;
	int batch_size = 1;

	std::vector<int> idx_vec(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		idx_vec[k] = k;
	}

	uint32_t seed = (uint32_t)(std::chrono::system_clock::now().time_since_epoch().count());
	std::mt19937_64 generator(seed);

	for (int c = 0; c < epoch; ++c)
	{
		std::shuffle(std::begin(idx_vec), std::end(idx_vec), generator);
		for (int i = 0; i < img_count / batch_size; ++i)
		{
			std::vector<VectorN> batch_img_vec(batch_size);
			std::vector<VectorN> batch_label_vec(batch_size);
			for (int k = 0; k < batch_size; ++k)
			{
				int j = idx_vec[(i * batch_size + k) % img_count];
				batch_img_vec[k] = img_vec[j];
				batch_label_vec[k] = lab_vec[j];
			}
			train(batch_img_vec, batch_label_vec, learning_rate);
		}

		uint32_t correct = test(test_img_vec, test_lab_vec);
		float correct_rate = (1.0f * correct / test_img_count);
		cout << "epoch " << c << ": " << correct_rate << " (" << correct << " / " << test_img_count << ")" << endl;
	}

	system("pause");
	return 0;

}

