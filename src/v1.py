
# coding: utf-8

# - **Training set**: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
# - **Test set**: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
# 
# # Aim
# Your goal is to build an algorithm capable of recognizing a sign with high accuracy. To do so, you are going to build a tensorflow model that is almost the same as one you have previously built in numpy for cat recognition (but now using a softmax output). It is a great occasion to compare your numpy implementation to the tensorflow one.
# 
# The model is `LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX`. The SIGMOID output layer has been converted to a SOFTMAX. A SOFTMAX layer generalizes SIGMOID to when there are more than two classes.
# 
# tf.nn.sigmoid_cross_entropy_with_logits, which computes
# $$- \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log \sigma(z^{[2](i)}) + (1-y^{(i)})\log (1-\sigma(z^{[2](i)})\large )\small\tag{2}$$

# In[175]:


import h5py
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import math
from PIL import Image
from scipy import ndimage

get_ipython().run_line_magic('matplotlib', 'inline')


# In[176]:


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
print("image shape: " + str(X_train_orig.shape))
print("label shape: " + str(Y_train_orig.shape))
# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# In[177]:


#As usual you flatten the image dataset, then normalize it by dividing by 255
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# In[178]:


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
        Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')

    return X, Y
# X, Y = create_placeholders(12288, 6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))


# In[179]:



# def initialize_parameters():
#     """
#     Initializes parameters to build a neural network with tensorflow. The shapes are:
#                         W1 : [25, 12288]
#                         b1 : [25, 1]
#                         W2 : [12, 25]
#                         b2 : [12, 1]
#                         W3 : [6, 12]
#                         b3 : [6, 1]
    
#     Returns:
#     parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
#     """
# #     tf.set_random_seed(1)    # "random" numbers same
#     #Returns an initializer performing "Xavier" initialization for weights.
#     W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
#     b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
#     W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
#     b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
#     W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
#     b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

#     parameters = {"W1": W1,
#               "b1": b1,
#               "W2": W2,
#               "b2": b2,
#               "W3": W3,
#               "b3": b3}
    
#     return parameters

# # tf.reset_default_graph()
# # with tf.Session() as sess:
# #     parameters = initialize_parameters()
# #     print("W1 = " + str(parameters["W1"]))
# #     print("b1 = " + str(parameters["b1"]))
# #     print("W2 = " + str(parameters["W2"]))
# #     print("b2 = " + str(parameters["b2"]))

# def forward_propagation(X, parameters):
#     """
#     Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
#     Arguments:
#     X -- input dataset placeholder, of shape (input size, number of examples)
#     parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
#                   the shapes are given in initialize_parameters

#     Returns:
#     Z3 -- the output of the last LINEAR unit
#     """
    
#     # Retrieve the parameters from the dictionary "parameters" 
#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
#     W3 = parameters['W3']
#     b3 = parameters['b3']
#     with tf.name_scope("layer1"):
#         Z1 = tf.add(tf.matmul(W1, X), b1)               # Z1 = np.dot(W1, X) + b1
#         A1 = tf.nn.relu(Z1)                             # A1 = relu(Z1)
#     with tf.name_scope("layer2"):
#         Z2 = tf.add(tf.matmul(W2, A1), b2)              # Z2 = np.dot(W2, a1) + b2
#         A2 = tf.nn.relu(Z2)                             # A2 = relu(Z2)
#     with tf.name_scope("layer3"):
#         Z3 = tf.add(tf.matmul(W3, A2), b3)              # Z3 = np.dot(W3,Z2) + b3
#     return Z3

# # tf.reset_default_graph()
# # with tf.Session() as sess:

# #     X, Y = create_placeholders(12288, 6)
# #     parameters = initialize_parameters()
# #     Z3 = forward_propagation(X, parameters)
# #     print("Z3 = " + str(Z3))


# In[180]:


# 整合一下

def nn_layer(input_tensor, input_dim, output_dim, layer_name, is_act=True, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[input_dim,1]))
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add(tf.matmul(weights, input_tensor), biases)
        if is_act:
            activations = act(preactivate, name='activation')  
        else:
            activations = preactivate
        return activations, weights, biases
    
def forward_propagation(X):
    hidden1, W1, b1 = nn_layer(X, 25, 12288, "layer1")
    hidden2, W2, b2 = nn_layer(hidden1, 12, 25, "layer2")
    output, W3, b3 = nn_layer(hidden2, 6, 12, "layer3", is_act=False)
    parameters = {"W1": W1,
          "b1": b1,
          "W2": W2,
          "b2": b2,
          "W3": W3,
          "b3": b3}
    return output, parameters


# In[181]:


def compute_cost(Z3, Y):
    """
    Computes the cost 
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    with tf.name_scope("loss_function"):
        #logits labels, expected to be of shape (number of examples, num_classes).
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        #tf.reduce_mean basically does the summation over the examples.
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print("cost = " + str(cost))


# In[182]:


#TODO：尝试不同改进方法：
#1.学习率衰减
#2.正则项
#3.滑动平均处理
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    X, Y = create_placeholders(n_x, n_y)      # Create Placeholders of shape (n_x, n_y)
    
    #自己改进一下
#     parameters = initialize_parameters()      # Initialize parameters
#     Z3 = forward_propagation(X, parameters)       # Forward propagation: Build the forward propagation in the tensorflow graph
    Y_hat, parameters = forward_propagation(X)
    cost = compute_cost(Y_hat, Y)       # Cost function: Add cost function to tensorflow graph
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)     # Backpropagation: Define the tensorflow optimizer.
    init = tf.global_variables_initializer()

    #write the computation graph to tensorboard log file
    writer = tf.summary.FileWriter("log/v1", tf.get_default_graph())
    writer.close()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Y_hat), tf.argmax(Y))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
if __name__ =='__main__':
    parameters = model(X_train, Y_train, X_test, Y_test)


# In[183]:


# #test with your own image
# my_image = "ourself_image.jpg"

# # We preprocess your image to fit your algorithm.
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
# my_image_prediction = predict(my_image, parameters)

# plt.imshow(image)
# print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

