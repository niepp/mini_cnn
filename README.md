# mini_nn

基于C++11的神经网络实验，通过造轮子加深对基本原理的理解

一个L层的神经网络，第1层为输入层，第L层为输出层，对第l层来说</br>
![image](https://github.com/niepp/mini_nn/blob/master/backprop.png)</br>

反向传播总结为:</br>
1.	根据训练样本给神经网络输入x,和输出y</br>
2.	从第2层到第L层，进行前向网络计算</br>
3.	根据公式（4）计算第L层的残差</br>
4.	从第L层到第2层，根据公式（3）逐级计算每层的残差，然后根据残差由公式（1）和（2）计算每层的梯度</br>


【实验1】 3层神经网络训练mnist数据集</br>
|---- 输入层(784 个神经元) </br>
|           </br>
|---- 隐含层(30 个神经元)</br>
|           </br>
|---- 输出层(10 个神经元)</br>
</br>
30 epoch, 10 batch size, 3.0 learning rate, the result on mnist dataset</br>

</br>
epoch 0: 0.9168 (9168 / 10000)</br>
epoch 1: 0.9297 (9297 / 10000)</br>
epoch 2: 0.9379 (9379 / 10000)</br>
epoch 3: 0.9386 (9386 / 10000)</br>
epoch 4: 0.942 (9420 / 10000)</br>
epoch 5: 0.9426 (9426 / 10000)</br>
epoch 6: 0.9438 (9438 / 10000)</br>
epoch 7: 0.9451 (9451 / 10000)</br>
epoch 8: 0.9458 (9458 / 10000)</br>
epoch 9: 0.9469 (9469 / 10000)</br>
epoch 10: 0.9487 (9487 / 10000)</br>
epoch 11: 0.9516 (9516 / 10000)</br>
epoch 12: 0.9493 (9493 / 10000)</br>
epoch 13: 0.9471 (9471 / 10000)</br>
epoch 14: 0.9486 (9486 / 10000)</br>
epoch 15: 0.9482 (9482 / 10000)</br>
epoch 16: 0.9501 (9501 / 10000)</br>
epoch 17: 0.9509 (9509 / 10000)</br>
epoch 18: 0.9515 (9515 / 10000)</br>
epoch 19: 0.9491 (9491 / 10000)</br>
epoch 20: 0.9516 (9516 / 10000)</br>
epoch 21: 0.9516 (9516 / 10000)</br>
epoch 22: 0.9494 (9494 / 10000)</br>
epoch 23: 0.9517 (9517 / 10000)</br>
epoch 24: 0.9515 (9515 / 10000)</br>
epoch 25: 0.9513 (9513 / 10000)</br>
epoch 26: 0.954 (9540 / 10000)</br>
epoch 27: 0.9502 (9502 / 10000)</br>
epoch 28: 0.955 (9550 / 10000)</br>
epoch 29: 0.9554 (9554 / 10000)</br>
maxCorrectRate: 0.9554</br>

[参考文献]</br>
1) [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by By Michael Nielsen</br>
2) [Deep Learning](http://www.deeplearningbook.org/), book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville</br>
3) http://cs231n.github.io/convolutional-networks </br>
4) http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf
