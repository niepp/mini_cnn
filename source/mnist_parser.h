#ifndef __MNIST_PARSER_H__
#define __MNIST_PARSER_H__

#include <string>
#include <fstream>

using namespace mini_cnn;

class mnist_parser
{
public:
	static const int N_inputCount = 784;
	static const int W_input = 28;
	static const int H_input = 28;
	static const int D_input = 1;
	static const int C_classCount = 10;
private:
	std::string m_relate_data_path;

public:
	mnist_parser(std::string relate_data_path)
		: m_relate_data_path(relate_data_path)
	{
	}

	void read_dataset(varray_vec &img_vec, varray_vec &lab_vec, varray_vec &test_img_vec, varray_vec &test_lab_vec)
	{
		// read train data
		// train images
		unsigned char *images = read_file(m_relate_data_path + "train-images.idx3-ubyte");
		int index = 0;
		int img_migic = read_int(images, index);
		int img_count = read_int(images, index);
		int col = read_int(images, index);
		int row = read_int(images, index);

		// train labels
		unsigned char *labels = read_file(m_relate_data_path + "train-labels.idx1-ubyte");
		int idx = 0;
		int lab_migic = read_int(labels, idx);
		int lab_count = read_int(labels, idx);

		nn_assert(img_count == lab_count);

		img_vec.resize(img_count);
		for (int k = 0; k < img_count; ++k)
		{
			img_vec[k] = new varray(N_inputCount);
			for (int i = 0; i < N_inputCount; ++i)
			{
				nn_float v = (nn_float)(images[index + k * N_inputCount + i] * 1.0 / 255.0);
				(*img_vec[k])[i] = v;
			}
		}

		lab_vec.resize(img_count);
		for (int k = 0; k < img_count; ++k)
		{
			lab_vec[k] = new varray(C_classCount);
			int lab = labels[idx + k];
			(*lab_vec[k])[lab] = (nn_float)1.0;
		}

		// read test data
		// test images
		unsigned char *test_images = read_file(m_relate_data_path + "t10k-images.idx3-ubyte");
		int test_idx = 0;
		int test_img_migic = read_int(test_images, test_idx);
		int test_img_count = read_int(test_images, test_idx);
		col = read_int(test_images, test_idx);
		row = read_int(test_images, test_idx);

		// test labels
		unsigned char *test_labels = read_file(m_relate_data_path + "t10k-labels.idx1-ubyte");
		int lab_idx = 0;
		int test_lab_migic = read_int(test_labels, lab_idx);
		int test_lab_count = read_int(test_labels, lab_idx);

		nn_assert(test_img_count == test_lab_count);

		test_img_vec.resize(test_img_count);
		for (int k = 0; k < test_img_count; ++k)
		{
			test_img_vec[k] = new varray(N_inputCount);
			for (int i = 0; i < N_inputCount; ++i)
			{
				nn_float v = (nn_float)(test_images[test_idx + k * N_inputCount + i] * 1.0 / 255.0);
				(*test_img_vec[k])[i] = v;
			}
		}

		test_lab_vec.resize(test_lab_count);
		for (int k = 0; k < test_lab_count; ++k)
		{
			test_lab_vec[k] = new varray(C_classCount);
			int lab = test_labels[lab_idx + k];
			(*test_lab_vec[k])[lab] = (nn_float)1.0;
		}
	}

private:
	int read_int(unsigned char *buffer, int &index)
	{
		int vint = (buffer[index] << 24) | (buffer[index + 1] << 16) |
			(buffer[index + 2] << 8) | (buffer[index + 3]);
		index += 4;
		return vint;
	}

	unsigned char* read_file(std::string file_path)
	{
		std::fstream fsread(file_path, std::fstream::in | std::fstream::binary);
		if (!fsread)
		{
			std::cerr << "Open failed!" << file_path << std::endl;
			fsread.exceptions(std::ios::failbit);
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

};

#endif // __MNIST_PARSER_H__
