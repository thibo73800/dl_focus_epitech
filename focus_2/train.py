from sframe import SFrame
import numpy as np
import random

def process_csv(csv):
    """
        Normalize and select usefull columns
        ** return (Tuple (numpy array, numpy array, numpy array)) **
            *train_features.shape : (n, 5)
            *train_targets.shape : (n)
            *test_features.shape : (n, 5)
            With n the numbers of rows
    """
    # Replace None value by 0
    csv = csv.fillna('Age', 0).fillna('Fare', 0)
    # Translate sex values with binary value (0 or 1)
    csv["sex"] = csv["Sex"].apply(lambda v : 1 if v == "male" else 0)
    # Normalize the age columns
    csv["age_normalize"] = (csv["Age"] - csv["Age"].mean()) /csv["Age"].std()
    # Normalize the faire columns
    csv["fare"] = (csv["Fare"] - csv["Fare"].mean()) /csv["Fare"].std()


    # We select the Survived columns and export it as a numpy array
    train_targets = csv[csv["type"] == "train"].select_columns(['Survived']).to_numpy()

    # We do the same for the features. For training features and for testing features
    train_features = csv[csv["type"] == "train"].select_columns(['sex', 'age_normalize', 'fare', 'SibSp', 'Parch']).to_numpy()
    test_features = csv[csv["type"] == "test"].select_columns(['sex', 'age_normalize', 'fare', 'SibSp', 'Parch']).to_numpy()

    return train_features, train_targets[:], test_features

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

def shuffle_training(features, targets):
    """
        Shuffle the two lists keeping the order
        ** input : **
            *features : numpy array of features
            *targets : numpy vector of targets
        ** return (numpy array of features, numpy vector of targets) **
    """
    c = list(zip(features.tolist(), targets.tolist()))
    random.shuffle(c)
    features[:], targets[:] = zip(*c)
    return np.array(features), np.array(targets)

if __name__ == '__main__':
    # Load both csv with sframe
    train_data = SFrame.read_csv("train.csv")
    test_data = SFrame.read_csv("test.csv")

    test_data["Survived"] = -1
    # We add a new columns for each csv to be abel to differentiate them later
    train_data["type"] = "train"
    test_data["type"] = "test"
    # We now can merge the two csv together
    data = train_data.append(test_data)

    # We extract features and targets from the csv
    train_features, train_targets, test_features = process_csv(data)

    # We initialize all variables. The weight is a one dimensional vector (one weight per feature)
    weights = np.random.randn(train_features.shape[1])
    # The bias
    b = 0
    # The learning rate
    lr = 0.5

    # We launch 200 epochs of optimization. One epoch consist of several different batch across all the training dataset
    for e in range(200):

        # Optional ! But in many case shuffling the data set is a good practise.
        train_features, train_targets = shuffle_training(train_features, train_targets)

        loss_list = [] # Use to store the average of loss for each epoch
        for features, targets in get_batches(train_features, train_targets, 50):

            # We initialize all gradient to zeros
            gradient_weights = np.zeros(weights.shape)
            gradient_b = 0
            py = [] # Use to store each model prediction for each batch

            for feature, target in zip(features, targets):
                # Compute z, the pre-activation neurons
                z = np.dot(feature, weights) + b
                # Compute y, the activation function
                y = sigmoid(z)
                # Append the prediction to the list
                py.append(y)

                # Update each gradient by taking the partial derviate of each variables.
                # We can do this in on line since we are working with numpy vector
                gradient_weights += -(target - y) * feature * derivate_sig(z)
                gradient_b += -(target - y) * derivate_sig(z)

            # Then we compute the mean of each gradient
            gradient_weights = gradient_weights / len(features)
            gradient_b = gradient_b / len(features)

            # Finally, we update each variables
            weights = weights - (lr * gradient_weights)
            b = b - (lr * gradient_b)

            # We add the current batch loss to the list of loss
            loss = (np.mean((targets - py) ** 2))
            loss_list.append(loss)

        # We now got a list of loss for each optimization for the current epoch
        # We take the average of the list and display it every 10 epochs.
        loss_mean = np.mean(loss_list)
        if e % 10 == 0:
            print "%s/200 loss_mean = %s" % (e, loss_mean)

    # The optimization is done, we can compute the prediction across all the dataset in one line
    train_py = [sigmoid(np.dot(feature, weights) + b) for feature in train_features]
    # The following line compute and display the full loss on the training dataset
    print "Final loss = ", np.mean((train_targets - train_py) ** 2)
    # The following line compute and display the accuracy  on the training dataset
    print "Final Accuracy = %s%%" % np.round((np.mean(train_targets == np.round(train_py)) * 100))

    # We no compute the prediction across all the testing dataset
    test_cls = np.round([sigmoid(np.dot(feature, weights) + b) for feature in test_features])[:]
    # We create the new columns in the csv with ours predictions
    test_data["Survived"] = [int(v) for v in test_cls]
    # We remove unused columns
    test_data.remove_columns(["type", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"])
    # Finally we can export this sframe into a new csv file which can be upload on kaggle
    test_data.save('prediction.csv', format='csv')
