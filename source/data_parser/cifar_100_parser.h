#ifndef __CIFAR_100_PARSER__
#define __CIFAR_100_PARSER__

#include <string>
#include <fstream>

using namespace mini_cnn;

class cifar_100_parser
{
public:
	static const int N_trainImgCount = 50000;
	static const int N_testImgCount = 10000;
	static const int W_img = 32;
	static const int H_img = 32;
	static const int D_img = 3;
	static const int Size_img = W_img * H_img * D_img;
	static const int C_classCount = 100;

private:
	std::string m_relate_data_path;

public:
	cifar_100_parser(std::string relate_data_path)
		: m_relate_data_path(relate_data_path)
	{
	}

	void read_data_batch(std::string data_batch_file, varray_vec &img_vec, varray_vec &lab_vec, int count)
	{
		unsigned char *buffer = nullptr;
		long size = read_file(m_relate_data_path + "train.bin", buffer);
		img_vec.resize(count);
		lab_vec.resize(count);
		int index = 0;
		for (int i = 0; i < count; ++i)
		{
			index++;
			// label
			lab_vec[i] = new varray(C_classCount);
			int lab = buffer[index++];
			(*lab_vec[i])[lab] = (nn_float)1.0;

			// image
			img_vec[i] = new varray(Size_img);
			for (int k = 0; k < Size_img; ++k)
			{
				nn_float v = (nn_float)(buffer[index++] * 1.0 / 255.0);
				(*img_vec[i])[k] = v;
			}
		}
	}

	void read_dataset(varray_vec &img_vec, varray_vec &lab_vec, varray_vec &test_img_vec, varray_vec &test_lab_vec)
	{
		// read train data
		read_data_batch("train.bin", img_vec, lab_vec, N_trainImgCount);

		// read test data
		read_data_batch("test.bin", test_img_vec, test_lab_vec, N_testImgCount);
	}

private:
	long read_file(std::string file_path, unsigned char *&buffer)
	{
		std::fstream fsread(file_path, std::fstream::in | std::fstream::binary);
		if (!fsread)
		{
			std::cerr << "Open failed!" << file_path << std::endl;
			fsread.exceptions(std::ios::failbit);
			return -1;
		}

		long size;

		//get length of file:
		fsread.seekg(0, std::ios::end);
		size = (long)fsread.tellg();
		fsread.seekg(0, std::ios::beg);

		buffer = new unsigned char[size];
		fsread.read(reinterpret_cast<char*>(buffer), size);
		fsread.close();

		return size;
	}

};

#endif // __CIFAR_100_PARSER__
