######################################################################
#                   Préparation des données
#                   Author :  Youssef Kartit
######################################################################

##### Libraries :
import csv
import random as rd

#### Charger le fichier CSV
def csv_2_list(filename, delimiter):
    ''' This funcion ''' 
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter = delimiter)
        dataset = list(csv_reader)
    return dataset

#### Convertire les éléments d'une colonne en flottants
def str_2_float(dataset, column):
    for i in range(len(dataset)):
        dataset[i][column] = float(dataset[i][column])
    return dataset

def str_2_int(dataset, column):
    classifications_values = [row[column] for row in dataset]
    unique_class_values = set(classifications_values)
    ind_val = {}
    for i, value in enumerate(unique_class_values):
        ind_val[value] = i
    for row in dataset:
        row[column] = ind_val[row[column]]
    return ind_val

#### Normalisation des données
def minmax_dataset(dataset):
    min_max = []
    for j in range(len(dataset[0])):
        column_values = [dataset[i][j] for i in range(len(dataset))]
        min_value, max_value = min(column_values), max(column_values)
        min_max.append([min_value, max_value])
    return min_max

def normalize_data(dataset):
    min_max = minmax_dataset(dataset)
    for i in range(len(dataset)):
        for j  in range(len(dataset[0])):
            dataset[i][j] = (dataset[i][j] - min_max[j][0]) / (min_max[j][1] - min_max[j][0])

#### Cross Validation en k folds
def crossvalidation_split(dataset, K):
    dataset_copy = list(dataset)
    dataset_cv = []
    fold_size = int(len(dataset) / K)
    for k in range(K):
        fold = []
        while len(fold) < fold_size:
            index = rd.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_cv.append(fold)
    return dataset_cv