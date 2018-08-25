#ifndef __MINIST_DATASET_H__
#define __MINIST_DATASET_H__

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <cassert>

#include "types.h"
#include "utils.h"
#include "math/vectorn.h"
#include "math/matrixmxn.h"
#include "math/matrix3d.h"
#include "math/mathdef.h"

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

#endif // __MINIST_DATASET_H__
