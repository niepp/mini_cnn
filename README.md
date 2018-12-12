
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
	- dropout layer
- activation functions
	- sigmoid
	- softmax
	- rectified linear(relu)
- loss functions
	- cross-entropy
	- mean squared error
- optimization algorithms
	- stochastic gradient descent
### Todo list
	- fast convolution(gemm, winograd)
	- train on gpu
	- batch normalization
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

std::mt19937_64 global_setting::m_rand_generator = std::mt19937_64(get_now_ms());

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

	// define neural network
	network nn = create_cnn();

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
			<< "  accuracy: " << cur_accuracy
			<< "  tot_cost: " << tot_cost 
			<< "  train elapse: " << train_elapse << "(s)" 
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

	nn_float max_accuracy = nn.SGD(img_vec, lab_vec, test_img_vec, test_lab_vec, epoch, batch_size, learning_rate, nthreads, minibatch_callback, epoch_callback);

	cout << "max_accuracy: " << max_accuracy << endl;

	auto t1 = get_now_ms();

	nn_float timeCost = (t1 - t0) * 0.001f;
	cout << "time_cost: " << timeCost << "(s)" << endl;

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
total paramters count:1668490</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 1/10  accuracy: 0.9852  train elapse: 860.249(s)  test elapse: 73.221(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 2/10  accuracy: 0.9881  train elapse: 893.153(s)  test elapse: 81.549(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 3/10  accuracy: 0.9897  train elapse: 897.536(s)  test elapse: 80.905(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 4/10  accuracy: 0.9909  train elapse: 882.907(s)  test elapse: 73.641(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 5/10  accuracy: 0.9911  train elapse: 889.855(s)  test elapse: 80.294(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 6/10  accuracy: 0.9912  train elapse: 867.259(s)  test elapse: 78.927(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 7/10  accuracy: 0.9907  train elapse: 864.71(s)  test elapse: 76.75(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 8/10  accuracy: 0.9919  train elapse: 841.899(s)  test elapse: 79.381(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 9/10  accuracy: 0.9917  train elapse: 843.356(s)  test elapse: 77.73(s)</br>
0%   10   20   30   40   50   60   70   80   90   100%</br>
|----|----|----|----|----|----|----|----|----|----|</br>
**************************************************</br>
epoch 10/10  accuracy: 0.993  train elapse: 838.781(s)  test elapse: 75.225(s)</br>
**max_accuracy: 0.993**</br>
time_cost: 9457.42(s)</br>


## References</br>
[1] [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by By Michael Nielsen</br>
[2] [Deep Learning](http://www.deeplearningbook.org/), book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville</br>
[3] http://cs231n.github.io/convolutional-networks </br>
[4] http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf</br>
[5] http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/</br>
[6] http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic</br>
[7] [Gradient checking](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)</br>
[8] [2D Max Pooling Backward Layer](https://software.intel.com/sites/products/documentation/doclib/daal/daal-user-and-reference-guides/daal_prog_guide/GUID-2C3AA967-AE6A-4162-84EB-93BE438E3A05.htm)</br>
[9] https://blog.csdn.net/mrhiuser/article/details/52672824

