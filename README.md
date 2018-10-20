
# **mini_cnn** is a light-weight convolutional neural network implementation based on c++11, mutli threading and head only


## Features</br>
- mutli threading
- gradient checking for all layer weights/bias
- weight initializer
	- xavier initialize
	- he initializer
- layer-types
	- fully connected layer
	- convolutional layer
	- softmax loglikelihood output layer
	- sigmod cross entropy output layer
	- average pooling layer
	- max pooling layer
- activation functions
	- sigmoid
	- softmax
	- rectified linear(relu)
- loss functions
	- cross-entropy
	- mean squared error
- optimization algorithms
	- stochastic gradient descent
### todo list
	- fast convolution(gemm, winograd)
	- train on gpu
	- batch normalization
	- dropout layer
	- more optimization algorithms such as adagrad，momentum etc	
	- serilize/deserilize
## Examples</br>
train **mnist** dataset</br>
```cpp
#include "mini_cnn.h"
#include "mnist_dataset_parser.h"

using namespace std;
using namespace mini_cnn;

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
	mnist_dataset_parser mnist(relate_data_path, "train-images-idx3-ubyte", "train-labels-idx1-ubyte"
								, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
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

	double learning_rate = 0.1;
	int epoch = 20;
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
```
## Result</br>

2-layer conv on mnist dataset</br>
```
	conv 3x3x32 relu
	   |
	maxpool 2x2
	   |
	conv 3x3x64 relu
	   |
	maxpool 2x2
	   |
	fc 1024
	   |
	log-likelihood softmax 10
```
about 98.8 correct rate

## References</br>
[1]  [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by By Michael Nielsen</br>
[2]  [Deep Learning](http://www.deeplearningbook.org/), book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville</br>
[3]  http://cs231n.github.io/convolutional-networks </br>
[4] http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf</br>
[5] http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/</br>
[6] http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic</br>
[7] [Gradient checking](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)</br>
[8] [2D Max Pooling Backward Layer](https://software.intel.com/sites/products/documentation/doclib/daal/daal-user-and-reference-guides/daal_prog_guide/GUID-2C3AA967-AE6A-4162-84EB-93BE438E3A05.htm)
[9] ["Fast Algorithms for Convolutional Neural Networks" Lavin and Gray, CVPR 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)

