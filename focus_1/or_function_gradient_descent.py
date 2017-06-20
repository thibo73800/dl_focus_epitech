"""
    Implementation of the gradient descent algorithm for
    only one neuron
    In this file the dataset is simply the true table of the OR function.
"""

def error_function(w1, w2, b, y, t):
    """
        @w1, @w2 : Neuron weight associated with each input
        @b : Neuron bias
        @y : neuron prediction
        @t : true target value
        Error function
    """
    return (1./2.) * (t - y)**2

if __name__ == '__main__':

    # Init neuron variables
    w1 = 0.1
    w2 = 0.5
    b = 0.3

    # Learning rate
    lr = 0.01

    # Dataset : True table of the OR function
    features = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [0, 1, 1, 1]

    for e in range(20): # 20 Epoch of iterations should be enougth to find the solution

        # Init all gradiant to zero
        gradiant_w1 = 0
        gradiant_w2 = 0
        gradiant_b = 0

        print "====================== e = ", e

        # Go through all pairs of inputs and targets
        for feature, target in zip(features, targets):
            # Compute the current neural prediction for those inputs
            y = ((w1 * feature[0]) + (w2 * feature[1])) + b

            # Print the current prediction
            print "Feature = ", feature, "Target = ", target, "Prediction = ", 1 if y > 0.5 else 0

            # Update gradients
            gradiant_w1 += -(target - y) * feature[0]
            gradiant_w2 += -(target - y) * feature[1]
            gradiant_b += -(target - y)

        # Update variables
        w1 = w1 - (lr * gradiant_w1)
        w2 = w2 - (lr * gradiant_w2)
        b = b - (lr * gradiant_b)
