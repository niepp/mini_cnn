#include <iostream>

#include "source/mini_cnn.h"
#include "source/mnist_parser.h"
#include "source/cifar_10_parser.h"
#include "source/cifar_100_parser.h"

using namespace std;
using namespace mini_cnn;

class progress_bar
{
	int m_tick_count;
	float m_cur_progress;
	int m_cur_ticks;
public:
	progress_bar(int tick = 50) : m_tick_count(tick), m_cur_progress(0), m_cur_ticks(0)
	{
	}

	void begin()
	{
		m_cur_progress = 0;
		m_cur_ticks = 0;
		std::cout << "0%   10   20   30   40   50   60   70   80   90   100%\n"
			<< "|----|----|----|----|----|----|----|----|----|----|"
			<< std::endl;
	}

	void grow(float cur_progress)
	{
		float dt = cur_progress - m_cur_progress;
		int d_ticks = static_cast<int>(dt * m_tick_count);
		if (d_ticks > 0)
		{
			m_cur_ticks += d_ticks;
			for (int i = 0; i < d_ticks; ++i)
			{
				cout << '*';
			}
			m_cur_progress = cur_progress;
		}

		if (1.0f - cur_progress < mini_cnn::cEpsilon)
		{
			for (int i = m_cur_ticks; i < m_tick_count; ++i)
			{
				cout << '*';
			}			
			cout << endl;
			m_cur_ticks = m_tick_count;
		}

	}

};

network create_mnist_fnn()
{
	network nn;
	nn.add_layer(new input_layer(mnist_parser::N_inputCount));
	nn.add_layer(new fully_connected_layer(100, activation_type::eIdentity));
	nn.add_layer(new relu_layer());

	nn.add_layer(new reshape_layer(10, 10, 1));

	nn.add_layer(new convolutional_layer(3, 3, 1, 32, 1, 1, padding_type::eValid));
	nn.add_layer(new relu_layer());

	nn.add_layer(new flatten_layer());

	nn.add_layer(new fully_connected_layer(30, activation_type::eIdentity));
	nn.add_layer(new relu_layer());
	nn.add_layer(new output_layer(mnist_parser::C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}

network create_mnist_cnn()
{
	network nn;
	nn.add_layer(new input_layer(mnist_parser::W_input, mnist_parser::H_input, mnist_parser::D_input));
	nn.add_layer(new convolutional_layer(3, 3, 1, 32, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
	nn.add_layer(new convolutional_layer(3, 3, 32, 64, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
	nn.add_layer(new fully_connected_layer(1024, activation_type::eRelu));
	nn.add_layer(new dropout_layer((nn_float)0.5));
	nn.add_layer(new output_layer(mnist_parser::C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}

network create_cifar_100_VGG16()
{
	network nn;
	nn.add_layer(new input_layer(cifar_100_parser::W_img, cifar_100_parser::H_img, cifar_100_parser::D_img));
	nn.add_layer(new convolutional_layer(3, 3, 3, 64, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 64, 64, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
	nn.add_layer(new convolutional_layer(3, 3, 64, 128, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 128, 128, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));

	nn.add_layer(new convolutional_layer(3, 3, 128, 256, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 256, 256, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 256, 256, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));

	nn.add_layer(new convolutional_layer(3, 3, 256, 512, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 512, 512, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 512, 512, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));

	nn.add_layer(new convolutional_layer(3, 3, 512, 512, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 512, 512, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new convolutional_layer(3, 3, 512, 512, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));

	nn.add_layer(new fully_connected_layer(4096, activation_type::eRelu));
	nn.add_layer(new fully_connected_layer(4096, activation_type::eRelu));
	nn.add_layer(new dropout_layer((nn_float)0.5));
	nn.add_layer(new output_layer(cifar_100_parser::C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}


network create_cifar_10_fnn()
{
	network nn;
	nn.add_layer(new input_layer(cifar_10_parser::Size_img));
	nn.add_layer(new fully_connected_layer(100, activation_type::eRelu));
	nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
	nn.add_layer(new output_layer(cifar_10_parser::C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}

//
//network create_cifar_10_cnn()
//{
//	network nn;
//	nn.add_layer(new input_layer(cifar_10_parser::W_img, cifar_10_parser::H_img, cifar_10_parser::D_img));
//	nn.add_layer(new convolutional_layer(3, 3, 3, 32, 1, 1, padding_type::eValid, activation_type::eRelu));
//	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
//	nn.add_layer(new convolutional_layer(3, 3, 32, 64, 1, 1, padding_type::eValid, activation_type::eRelu));
//	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
//	nn.add_layer(new fully_connected_layer(1024, activation_type::eRelu));
//	nn.add_layer(new dropout_layer((nn_float)0.5));
//	nn.add_layer(new output_layer(cifar_10_parser::C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
//	return nn;
//}


// random init
// seed = 2572007265;	// fixed seed to repeat test
std::mt19937_64 global_setting::m_rand_generator = std::mt19937_64(get_now_ms());

int main()
{
	varray_vec img_vec;
	varray_vec lab_vec;
	varray_vec test_img_vec;
	varray_vec test_lab_vec;

	mnist_parser mnist("../../dataset/mnist/");
	mnist.read_dataset(img_vec, lab_vec, test_img_vec, test_lab_vec);

	//cifar_10_parser cifar_10("../../dataset/cifar-10/");
	//cifar_10.read_dataset(img_vec, lab_vec, test_img_vec, test_lab_vec);

	//cifar_100_parser cifar_100("../../dataset/cifar-100/");
	//cifar_100.read_dataset(img_vec, lab_vec, test_img_vec, test_lab_vec);

	// define neural network
	network nn = create_mnist_cnn();
	//network nn = create_cifar_100_VGG16();

	cout << "total paramters count:" << nn.paramters_count() << endl;

	progress_bar train_progress_bar;
	train_progress_bar.begin();

	nn.init_all_weight(he_normal_initializer());

	float learning_rate = 0.1f;
	int epoch = 10;
	int batch_size = 10;
	nn_int nthreads = std::thread::hardware_concurrency();
	nthreads = 12;

	auto epoch_callback = [&train_progress_bar](nn_int c, nn_int epoch, nn_float cur_accuracy, nn_float tot_cost, nn_float train_elapse, nn_float test_elapse)
	{
		std::cout << "epoch " << c << "/" << epoch
			<< "  accuracy: " << cur_accuracy;
		if (tot_cost >= 0)
		{
			std::cout << "  tot_cost: " << tot_cost;
		}
		std::cout << "  train elapse: " << train_elapse << "(s)"
			<< "  test elapse: " << test_elapse << "(s)" << std::endl;
		if (c < epoch)
		{
			train_progress_bar.begin();
		}
	};

	auto minibatch_callback = [&train_progress_bar](nn_int cur_size, nn_int img_count)
	{
		train_progress_bar.grow(cur_size * 1.0f / img_count);
	};

	auto t0 = get_now_ms();

	nn_float max_accuracy = nn.SGD(img_vec, lab_vec, test_img_vec, test_lab_vec, epoch, batch_size, learning_rate, false, nthreads, minibatch_callback, epoch_callback);

	cout << "max_accuracy: " << max_accuracy << endl;

	auto t1 = get_now_ms();

	nn_float timeCost = (t1 - t0) * 0.001f;
	cout << "time_cost: " << timeCost << "(s)" << endl;

	system("pause");
	return 0;

}


