#ifndef __VOC2007_PARSER_H__
#define __VOC2007_PARSER_H__

#include <vector>
#include <string>
#include <fstream>

using namespace mini_cnn;

static const std::string cClass[] = {
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
	"car", "cat", "chair", "cow", "diningtable", "dog", "horse",
	"motorbike", "person", "pottedplant", "sheep", "sofa",
	"train", "tvmonitor" };

//dataset\voc2007\ImageSets\Main
//test.txt是测试集，train.txt是训练集，val.txt是验证集，trainval.txt是训练和验证集

class voc2007_parser
{
public:
	static const int IMAGE_SIZE = 448;
	static const int CELL_SIZE = 7;
	static const int	BOXES_PER_CELL = 2;
	//static const float	ALPHA = 0.1f;
	//static const float OBJECT_SCALE = 1.0f;
	//static const float NOOBJECT_SCALE = 1.0f;
	//static const float CLASS_SCALE = 2.0f;
	//static const float COORD_SCALE = 5.0f;

private:
	std::string m_relate_data_path;
	std::vector<std::string> m_train_image_names;
	std::vector<std::string> m_test_image_names;

public:
	voc2007_parser(std::string relate_data_path)
		: m_relate_data_path(relate_data_path)
	{
	}

	void read_dataset(varray_vec &img_vec, varray_vec &lab_vec, varray_vec &test_img_vec, varray_vec &test_lab_vec)
	{
		read_file(m_relate_data_path + "trainval/ImageSets/Main/trainval.txt", m_train_image_names);
		read_file(m_relate_data_path + "test/ImageSets/Main/test.txt", m_train_image_names);

		// read images

		// read labels
	
	}

private:
	void read_file(std::string file_path, std::vector<std::string> &image_names)
	{
		std::fstream fsread(file_path, std::fstream::in);
		if (!fsread)
		{
			std::cerr << "Open failed!" << file_path << std::endl;
			fsread.exceptions(std::ios::failbit);
			return;
		}

		std::string s;
		while (fsread >> s)
		{
			image_names.push_back(s);
		}
		fsread.close();
	}


};

#endif // __VOC2007_PARSER_H__
