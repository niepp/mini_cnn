
**mini_cnn** is a light-weighted convolutional neural network implementation based on c++11, mutli threading and head only


## Features</br>
### done</br>
- mutli threading
- gradient checking
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
	- batch normalization
	- dropout layer
	- more optimization algorithms such as adagrad，momentum etc
	- train on gpu
	- serilize/deserilize
## Examples</br>
train **fashion mnist** dataset</br>
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

	mnist_dataset_parser fashion("data/fashion/train-images-idx3-ubyte", "data/fashion/train-labels-idx1-ubyte"
								, "data/fashion/t10k-images-idx3-ubyte", "data/fashion/t10k-labels-idx1-ubyte");
	fashion.read_dataset(img_vec, lab_vec, test_img_vec, test_lab_vec);

	int img_count = img_vec.size();
	int test_img_count = test_img_vec.size();

	// random init
	std::mt19937_64 generator(get_now());

	// define neural network
	network nn = create_cnn();

	he_normal_initializer initializer(generator);

	nn.init_all_weight(initializer);

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
	nn.set_task_count(nthreads);

	for (int c = 0; c < epoch; ++c)
	{
		double minCost = cMAX_FLOAT;
		std::shuffle(idx_vec.begin(), idx_vec.end(), generator);
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

	return 0;

}
```

## References</br>
[1]  [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by By Michael Nielsen</br>
[2]  [Deep Learning](http://www.deeplearningbook.org/), book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville</br>
[3]  http://cs231n.github.io/convolutional-networks </br>
[4] http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf</br>
[5] http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/</br>
[6] http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic</br>
[7] [Gradient checking](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)</br>
[8] [2D Max Pooling Backward Layer](https://software.intel.com/sites/products/documentation/doclib/daal/daal-user-and-reference-guides/daal_prog_guide/GUID-2C3AA967-AE6A-4162-84EB-93BE438E3A05.htm)

