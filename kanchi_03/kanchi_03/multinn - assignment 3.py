# Kanchi, Gowtham Kumar
# 1002-044-003
# 2022_10_30
# Assignment_03_01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.transfer_func = []
        self.weights = []
        self.biases = []
        self.count_of_layers = 0
        self.layer_dimensions = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        self.layer_dimensions.append(num_nodes)
        if self.count_of_layers == 0:
            self.transfer_func.append(transfer_function)
            self.weights.append( np.random.randn(self.input_dimension,num_nodes))
            self.biases.append( np.random.randn(num_nodes,1))
        else:
            self.transfer_func.append(transfer_function)
            self.weights.append( np.random.randn( self.layer_dimensions[self.count_of_layers-1],num_nodes))
            self.biases.append(np.random.randn(num_nodes, 1))
        self.count_of_layers = self.count_of_layers + 1


    def get_weights_without_biases(self, layer_number):
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        
        self.biases[layer_number] = biases

    def calculate_loss(self, y, y_hat):
        
        loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=y_hat))
        return loss

    def predict(self, X):
   
        y = tf.Variable(X)
        for (w , b, transfer_func) in zip( self.weights ,self.biases,self.transfer_func):
            Z = tf.add(tf.matmul(y,w), tf.transpose(b))

            if transfer_func == "Sigmoid":
                y = tf.math.sigmoid(Z)

            elif transfer_func == "Linear":
                y = Z

            elif transfer_func == "Relu":
                y = tf.nn.relu(Z)

        return y

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        
        X_Train = tf.Variable(X_train)
        Y_Train = tf.Variable(y_train)
        for i in range(num_epochs):
            for j in range(0, np.shape(X_Train)[0], batch_size):
                val = j + batch_size
                X_Batch = X_Train[j:val, :]
                y_Batch = Y_Train[j:val]
                with tf.GradientTape() as tape:
                    predictions = self.predict(X_Batch)
                    loss = self.calculate_loss(y_Batch, predictions)
                    dloss_weights, dloss_bias = tape.gradient(loss, [self.weights, self.biases])

                for i in range(self.count_of_layers):
                    self.weights[i].assign_sub(alpha * dloss_weights[i])
                    self.biases[i].assign_sub(alpha * dloss_bias[i])



    def calculate_percent_error(self, X, y):
        
        x = self.predict(X)
        number_of_samples = np.shape(y)[0]
        error = y - np.argmax(x,axis=1)
        return(np.count_nonzero(error) / number_of_samples)

    def calculate_confusion_matrix(self, X, y):
       
        return tf.math.confusion_matrix(y,np.argmax(self.predict(X),axis=1))
