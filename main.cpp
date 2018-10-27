#include <iostream>

#include "source/mini_cnn.h"
#include "source/mnist_dataset_parser.h"

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

network create_fnn()
{
	network nn;
	nn.add_layer(new input_layer(N_inputCount));
	nn.add_layer(new fully_connected_layer(100, activation_type::eRelu));
	nn.add_layer(new fully_connected_layer(30, activation_type::eRelu));
	nn.add_layer(new output_layer(C_classCount, lossfunc_type::eSoftMax_LogLikelihood, activation_type::eSoftMax));
	return nn;
}

network create_cnn_small()
{
	network nn;
	nn.add_layer(new input_layer(W_input, H_input, D_input));
	nn.add_layer(new convolutional_layer(3, 3, 1, 4, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
	nn.add_layer(new convolutional_layer(3, 3, 4, 8, 1, 1, padding_type::eValid, activation_type::eRelu));
	nn.add_layer(new max_pooling_layer(2, 2, 2, 2));
	nn.add_layer(new fully_connected_layer(32, activation_type::eRelu));
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

	// random init
	nn_int seed = (nn_int)get_now_ms();
	seed = 2572007265;	// fixed seed to repeat test
	cout << "random seed:" << seed << endl;
	std::mt19937_64 generator(seed);

	// define neural network
	network nn = create_cnn();

	cout << "total paramters count:" << nn.paramters_count() << endl;

	progress_bar train_progress_bar;
	train_progress_bar.begin();

	//truncated_normal_initializer initializer(generator, 0, 0.1, 2);
	he_normal_initializer initializer(generator);

	nn.init_all_weight(initializer);

	float learning_rate = 0.1f;
	int epoch = 10;
	int batch_size = 100;
	nn_int nthreads = std::thread::hardware_concurrency();
	nthreads = 1;

	auto epoch_callback = [&train_progress_bar](nn_int c, nn_int epoch, nn_float cur_accuracy, nn_float tot_cost, nn_float elapse)
	{
		std::cout << "epoch " << c << "/" << epoch << "  accuracy: " << cur_accuracy << "  tot_cost: " << tot_cost << "  train elapse: " << elapse << "(s)" << std::endl;
		if (c < epoch)
		{
			train_progress_bar.begin();
		}
	};

	auto minibatch_callback = [&train_progress_bar](nn_int cur_size, nn_int img_count)
	{
		train_progress_bar.grow(cur_size * 1.0f / img_count);
	};

	nn_float t0 = get_now_ms();

	nn_float max_accuracy = nn.SGD(img_vec, lab_vec, test_img_vec, test_lab_vec, generator, epoch, batch_size, learning_rate, nthreads, minibatch_callback, epoch_callback);

	cout << "max_accuracy: " << max_accuracy << endl;

	nn_float t1 = get_now_ms();

	nn_float timeCost = (t1 - t0) * 0.001f;
	cout << "TimeCost: " << timeCost << "(s)" << endl;

	cout << "g_cost: " << g_cost << "(s)" << endl;
	cout << "g_conv_cnt: " << g_conv_cnt << endl;
	for (auto &p : g_shape_map)
	{
		auto s = p.first;
		cout << s.iw << ", " << s.ih << "\n" << s.fw << ", " << s.fh << "\n" << s.ow << ", " << s.oh << "\n--------" << p.second.cnt << "\t" << p.second.cost << "--------\n" << endl;
	}

	system("pause");
	return 0;

}


