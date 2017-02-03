## Synopsis

The program implements neural network with hidden layers. The activation function of output layer is softmax and the activation of hidden layers can be logistic or tanh. The backpropagation process can use mini_batch gradient descent method or momentum method. Besides, this program can also check the computation of gradient descent using numerical approximation.



## Code Example
####1.When changing the structure of the neural network, we can create a new instance of NeuralNet class in the main() function.
**Example:**<br />
create a neural network with one hidden layer having 800 nodes and using tanh as activation function, the learning rate is 0.01:<br />
```python
net = NeuralNet([784, 800, 10], 'tanh', 0.01)
```

####2.Start the training process:<br />
**Example:**<br />
Train neural network using 30 iterations and mini_batch with size 30, with backpropagation using momentum method, the learning rate for momentum term is 0.01<br />
```python
net.train(train_feature, train_label, val_feature, val_label, test_feature,test_label,n_iter=30, b_size=30, lamb=0.01, momentum=True,test_weight=False)
```
####3.Check the gradient descent computation by using numerical approximation:
**Example:**<br />
Make sure the train_feature has only one data.<br />
```python
  net.train(train_feature, train_label, val_feature, val_label, test_feature,test_label,n_iter=1, b_size=1, lamb=0.01,       momentum=False,test_weight=True)
```
## Contributors

Luning Yang, Bodong Zhang
