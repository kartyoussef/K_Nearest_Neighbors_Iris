######################################################################
#              Implémentation de l'Algorithme k-NN
#                   Author :  Youssef Kartit
######################################################################


####################################

from math import sqrt
import dataprep as dp

### The euclidian function between two vectors:
def euclidean_dist(x_1, x_2):
    distance = 0
    for i in range(len(x_1)-1):
        distance += (x_1[i] - x_2[i])**2
    return sqrt(distance)

### Locate k nearest neighbors
def nearest_neighbors(train, row_, k):
    distances = []
    for row in train:
        d = euclidean_dist(row, row_)
        distances.append((row, d))
    distances.sort(key = lambda tup : tup[1])
    neighbors = []
    for l in range(k):
        neighbors.append(distances[l][0])
    return neighbors

### Make a classification prediction with nearest neighbors
def predict_class(train, test_row, k):
    neighbors = nearest_neighbors(train, test_row, k)
    outputs = [row[-1] for row in neighbors]
    prediction = max(set(outputs), key = outputs.count)
    return prediction

### Accuracy pourcentage : 
def accuracy_metric(test, predicted):
    correct = 0
    for l in range(len(test)):
        if test[l] == predicted[l]:
            correct += 1
    return 100 * (correct / len(test))

#### Evaluer un algorithme :
def evaluate_algorithm(dataset, algorithm, K, *args):
    #create folds
    folds = dp.crossvalidation_split(dataset, K)
    #Initializing scores and r_sqr
    scores = []
    for fold in folds: 
        trainset = list(folds)
        trainset.remove(fold)
        trainset = sum(trainset, [])
        testset = []
        for row in fold:
            row_copy = list(row)
            testset.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(trainset, testset, *args)
        test = [row[-1] for row in fold]
        # Accuracy & Scores
        accuracy = accuracy_metric(test, predicted)
        scores.append(accuracy)
    return scores

### k-NN Algorithm : 
def k_NN(train, test, k):
    predictions = []
    for row in test : 
        pred = predict_class(train, row, k)
        predictions.append(pred)
    return predictions