#ifndef __CIFAR_10_PARSER__
#define __CIFAR_10_PARSER__

#include <string>
#include <fstream>

using namespace mini_cnn;

class cifar_10_parser
{
public:
	static const int N_imgBatchSize = 10000;
	static const int W_img = 32;
	static const int H_img = 32;
	static const int D_img = 3;
	static const int Size_img = W_img * H_img * D_img;
	static const int C_classCount = 10;

private:
	std::string m_relate_data_path;

public:
	cifar_10_parser(std::string relate_data_path)
		: m_relate_data_path(relate_data_path)
	{
	}

	void read_data_batch(std::string data_batch_file, varray_vec &img_vec, varray_vec &lab_vec)
	{
		unsigned char *buffer = nullptr;
		long size = read_file(m_relate_data_path + data_batch_file, buffer);
		int index = 0;
		int base_vec_index = img_vec.size();
		img_vec.resize(base_vec_index + N_imgBatchSize);
		lab_vec.resize(base_vec_index + N_imgBatchSize);

		for (int i = 0; i < N_imgBatchSize; ++i)
		{
			// label
			lab_vec[base_vec_index + i] = new varray(C_classCount);
			int lab = buffer[index++];
			(*lab_vec[base_vec_index + i])[lab] = (nn_float)1.0;

			// image
			img_vec[base_vec_index + i] = new varray(Size_img);
			for (int k = 0; k < Size_img; ++k)
			{
				nn_float v = (nn_float)(buffer[index + k] * 1.0 / 255.0);
				(*img_vec[base_vec_index + i])[k] = v;
			}
			index += Size_img;
		}
	}

	void read_dataset(varray_vec &img_vec, varray_vec &lab_vec, varray_vec &test_img_vec, varray_vec &test_lab_vec)
	{
		// read train data
		read_data_batch("data_batch_1.bin", img_vec, lab_vec);
		read_data_batch("data_batch_2.bin", img_vec, lab_vec);
		read_data_batch("data_batch_3.bin", img_vec, lab_vec);
		read_data_batch("data_batch_4.bin", img_vec, lab_vec);
		read_data_batch("data_batch_5.bin", img_vec, lab_vec);

		// read test data
		read_data_batch("test_batch.bin", test_img_vec, test_lab_vec);
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

#endif // __CIFAR_10_PARSER__
