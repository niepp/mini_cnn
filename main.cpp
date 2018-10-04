#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <future>
#include <array>

#include "mini_cnn.h"
#include "minist_dataset.h"

using namespace std;
using namespace mini_cnn;

std::mt19937_64 create_random()
{
	// random init
	uint_t seed = get_now();

	cout << "random seed:" << seed << endl;

	std::mt19937_64 generator(seed);

	return generator;
}

network create_fnn()
{
	network nn;
	nn.add_layer(new input_layer(N_inputCount));
	nn.add_layer(new fully_connected_layer(100, activation_type::eRelu));
	nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
	nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}

network create_fnn1()
{
	network nn;
	nn.add_layer(new input_layer(N_inputCount));
//	nn.add_layer(new fully_connected_layer(100, activation_type::eSigmod));
	nn.add_layer(new fully_connected_layer(30, activation_type::eSigmod));
	nn.add_layer(new output_layer(C_classCount, lossfunc_type::eMSE, activation_type::eSigmod));
	return nn;
}

int main()
{
	varray_vec img_vec;
	varray_vec lab_vec;
	varray_vec test_img_vec;
	index_vec test_lab_vec;
	read_dataset(img_vec, lab_vec, test_img_vec, test_lab_vec);

	int img_count = img_vec.size();
	int test_img_count = test_img_vec.size();

	long long t0 = get_now();
	auto generator = create_random();
	normal_random nrand(generator, 0, 0.1, 2);

	// define neural network
	network nn = create_fnn();

	nn.init_all_weight(nrand);

	double learning_rate = 0.1;
	int epoch = 20;
	int batch_size = 10;
	int batch = img_count / batch_size;
	std::vector<int> idx_vec(img_count);
	for (int k = 0; k < img_count; ++k)
	{
		idx_vec[k] = k;
	}

	double maxCorrectRate = 0;

	int nthreads = std::thread::hardware_concurrency();
	//nthreads = 1;
	nn.set_task_count(nthreads);

	for (int c = 0; c < epoch; ++c)
	{
		double minCost = cMAX_FLOAT;
		std::shuffle(std::begin(idx_vec), std::end(idx_vec), generator);
		varray_vec batch_img_vec(batch_size);
		varray_vec batch_label_vec(batch_size);
		for (int i = 0; i < batch; ++i)
		{
			for (int k = 0; k < batch_size; ++k)
			{
				int j = idx_vec[(i * batch_size + k) % img_count];
				batch_img_vec[k] = img_vec[j];
				batch_label_vec[k] = lab_vec[j];
			}

			nn.train(batch_img_vec, batch_label_vec, learning_rate, nthreads);

			if (i % (batch / 4) == 0)
			{
				double ca = nn.get_cost(batch_img_vec, batch_label_vec, nthreads);

				//if (ca < minCost)
				//{
				//	minCost = ca;
				//	int_t correct = nn.test(test_img_vec, test_lab_vec, nthreads);
				//	double correct_rate = (1.0 * correct / test_img_count);
				//	std::cout << "batch: " << i << "/" << batch << "  learning_rate:" << learning_rate << "  cost: " << ca << "  correct_rate: " <<
				//		correct_rate << " (" << correct << " / " << test_img_count << ")" << endl;
				//}
				//else
				{
					std::cout << "batch: " << i << "/" << batch << "  learning_rate:" << learning_rate << "  cost: " << ca << endl;
				}
			}

		}

		int_t correct = nn.test(test_img_vec, test_lab_vec, nthreads);
		double correct_rate = (1.0 * correct / test_img_count);
		if (correct_rate > maxCorrectRate)
		{
			maxCorrectRate = correct_rate;
		}

		double tot_cost = nn.get_cost(img_vec, lab_vec, nthreads);
		std::cout << "epoch " << c << ": " << correct_rate << " (" << correct << " / " << test_img_count << ")" << "  tot_cost = " << tot_cost << endl;

	}

	cout << "Max CorrectRate: " << maxCorrectRate << endl;

	long long t1 = get_now();

	float timeCost = (t1 - t0) * 0.001f;
	cout << "TimeCost: " << timeCost << "(s)" << endl;

	system("pause");
	return 0;

}


