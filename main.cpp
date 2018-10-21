#include <iostream>

#include "source/mini_cnn.h"
#include "source/mnist_dataset_parser.h"

using namespace std;
using namespace mini_cnn;

network create_fnn()
{
	network nn;
	nn.add_layer(new input_layer(N_inputCount));
	nn.add_layer(new fully_connected_layer(100, activation_type::eRelu));
	nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
	nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}

network create_cnn()
{
	network nn;
	nn.add_layer(new input_layer(W_input, H_input, D_input));
	nn.add_layer(new convolutional_layer(3, 3, 1, 32, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
	nn.add_layer(new convolutional_layer(3, 3, 32, 64, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
	nn.add_layer(new fully_connected_layer(1024, activation_type::eRelu));
	nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}

int main()
{
	varray_vec img_vec;
	varray_vec lab_vec;
	varray_vec test_img_vec;
	index_vec test_lab_vec;

	std::string relate_data_path = "../../dataset/mnist/";
	mnist_dataset_parser mnist(relate_data_path, "train-images.idx3-ubyte", "train-labels.idx1-ubyte"
		, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	mnist.read_dataset(img_vec, lab_vec, test_img_vec, test_lab_vec);

	uint_t t0 = get_now();

	// random init
	uint_t seed = t0;
	seed = 2572007265;	// fixed seed to repeat test
	cout << "random seed:" << seed << endl;
	std::mt19937_64 generator(seed);

	// define neural network
	network nn = create_cnn();

	cout << "total paramters count:" << nn.paramters_count() << endl;

	//truncated_normal_initializer initializer(generator, 0, 0.1, 2);
	he_normal_initializer initializer(generator);

	nn.init_all_weight(initializer);

	float learning_rate = 0.1f;
	int epoch = 10;
	int batch_size = 10;

	auto epoch_callback = [](int_t c, mini_cnn::float_t cur_accuracy, mini_cnn::float_t tot_cost) {
		std::cout << "epoch " << c << ": " << cur_accuracy << "  tot_cost = " << tot_cost << std::endl;
	};

	auto max_accuracy = nn.SGD(img_vec, lab_vec, test_img_vec, test_lab_vec, generator, epoch, batch_size, learning_rate, epoch_callback);

	cout << "max_accuracy: " << max_accuracy << endl;

	uint_t t1 = get_now();

	float timeCost = (t1 - t0) * 0.001f;
	cout << "TimeCost: " << timeCost << "(s)" << endl;

	system("pause");
	return 0;

}


