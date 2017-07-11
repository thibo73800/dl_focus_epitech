from sframe import SFrame
import numpy as np
import random

def get_batches(features, targets, batch_size):
    """
        Return new batches
        ** input : **
            *features : numpy array of features
            *targets : numpy vector of targets
            *batch_size : size of one batch
        **return (Python Generator) **
    """
    nb_batchs = len(features) // batch_size
    for b in range(0, nb_batchs * batch_size, batch_size):
        yield features[b : b + batch_size], targets[b : b + batch_size]

def columns_to_category(csv, columns):
    """
        Convert each none linear columns to one hot vector
        **input : **
            *csv : Sframe
            *columns : (List of string) List of non linear columns to convert
        ** return (Sframe) **
            Return a new Sframe with new columns
    """
    for column in columns:
        size = len(csv[column].unique())
        class_to_int = {c: i for i, c in enumerate(set(csv[column].unique()))}
        csv[column] = csv[column].apply(lambda x : np.eye(size)[class_to_int[x]] )

    return csv

def columns_to_normalize(csv, columns):
    """
        Convert each linear columns to one unique normalize value
        **input : **
            *csv : Sframe
            *columns : (List of string) List of linear columns to convert
        ** return (Sframe) **
            Return a new Sframe with new columns
    """
    for column in columns:
        csv[column] = (csv[column] - csv[column].mean()) /csv[column].std()
    return csv


def derivate_sig(z):
    """
        Compute the derviate of the sigmoid function
        ** input : **
            *z : pre-activation function
        ** return (Float value) **
    """
    return sigmoid(z) * (1 - sigmoid(z))

def sigmoid(z):
    """
        Compute the sigmoid function
        ** input : **
            *z : pre-activation function
        ** return (Float value) from O to 1  **
    """
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    dataset = SFrame.read_csv("adult.csv")

    CATEGORY_KEYS = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country"]
    CONTINU_KEYS = ["capital-gain", "fnlwgt", "hours-per-week", "age", "capital-loss", "educational-num"]

    # Process nonlinear columns
    dataset = columns_to_category(dataset, CATEGORY_KEYS)
    # Process linear columns
    dataset = columns_to_normalize(dataset, CONTINU_KEYS)
    # Convert the output from string to binary
    dataset["income"] = dataset["income"].apply(lambda x : 1. if x == ">50K" else 0.)

    keys = CATEGORY_KEYS + CONTINU_KEYS + ["income"]
    features = []
    # Create the features matrix
    for line in dataset:
        n_feature = []
        for key in keys:
            n_feature += [line[key]] if isinstance(line[key], float) else line[key]
        features.append(n_feature)

    features = np.array(features)
    # Extract all targets from features matrix
    targets = features[:,-1][:]
    # Remove targets from features matrix
    features = features[:,:-1]

    # Hyperparameters
    learning_step = 0.01
    epoch = 100
    hidden_layer_size = 10

    # Init weights and bais
    w1 = np.random.randn(features.shape[1], hidden_layer_size)
    w2 = np.random.randn(hidden_layer_size)
    b1 = np.zeros(hidden_layer_size)
    b2 = np.zeros(1)

    # We launch 100000 epochs of optimization. One epoch consist of several different batch across all the training dataset
    for e in xrange(epoch):
        i = 0

        mean_cost = [] # Use to store the loss average for each epoch
        mean_acc = [] # Use to store the accuracy average for each epoch

        for batch_features, batch_targets in get_batches(features, targets, 100):

            # We initialize all gradient to zeros
            g_w1 = np.zeros(w1.shape)
            g_w2 = np.zeros(w2.shape)
            g_b1 = np.zeros(b1.shape)
            g_b2 = np.zeros(b2.shape)

            for feature, target in zip(batch_features, batch_targets):
                # Compute z1, the pre-activation function for all neurons of the first layer
                z1 = np.dot(feature, w1) + b1
                # Compute z1, the activation function for all neurons of the first layer
                a1 = sigmoid(z1)
                # Compute z2, the pre-activation function for the output neuron
                z2 = np.dot(a1, w2) + b2
                # Compute the output (the probability)
                output = sigmoid(z2)

                # Compute the error term
                error = -1 * (target - output)
                # Compute the error term for the output layer (the single neuron)
                error_term_output = error * derivate_sig(z2)
                # Compute the error term fro the hidden layer
                error_term_hidden_layer = error_term_output * w2 * derivate_sig(z1)
                # Update gradients of the first layer
                g_w1 += error_term_hidden_layer * feature[:,None]
                g_w2 += error_term_hidden_layer * a1
                # Update gradients of the output neuron
                g_b1 += error_term_output * derivate_sig(z1)
                g_b2 += error_term_output

            # Apply gradient descent
            b1 -= g_b1 * learning_step
            b2 -= g_b2 * learning_step
            w1 -= g_w1 * learning_step
            w2 -= g_w2 * learning_step

            # Compute the predictions for all the dataset
            z1 = np.dot(batch_features, w1) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, w2) + b2
            outputs = sigmoid(z2)

            # Compute error metrics
            cost = np.mean((batch_targets - outputs) ** 2)
            acc = np.mean(batch_targets == np.round(outputs))
            mean_cost.append(cost)
            mean_acc.append(acc)

        print "---------- e = ", e
        print "Mean cost", np.mean(mean_cost)
        print "Mean acc", np.mean(mean_acc)
