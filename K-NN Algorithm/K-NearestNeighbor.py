import math
from random import seed
from random import randrange
from csv import reader
from unittest import result
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
# Load a csv file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
        
# Convert string column to float column
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        
# Convert string column to integer 
def str_column_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    unique = set(class_value)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def SelectionSort(row):
    for i in range(len(row)):
        minPosition = i
        min = row[i][1]
        for j in range(i + 1,len(row)):
            if row[j][1] <= min:
                min = row[j][1]
                minPosition = j
        temp = row[i]
        row[i] = row[minPosition]
        row[minPosition] = temp  


#compute Euclidean Distance
#row: data point
def EuclideanDistance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance = distance + (row1[i] - row2[i])**2
    return math.sqrt(distance)

#Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = EuclideanDistance(train_row, test_row)
        distances.append((train_row, dist))
        #print('{} \n'.format(distances))
    #return distances
    SelectionSort(distances)    
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    #print(neighbors)
    return neighbors 
    
   
def MakePrediction(training_set, test_set, num_neighbors):
    prediction = list()
    for test_point in test_set:
        neighbors = get_neighbors(training_set, test_point, num_neighbors)
        for row in neighbors:
            output_value = [row[-1]]
        predic = max(output_value, key=output_value.count)
        prediction.append(int(predic))
        neighbors.clear()
    prediction = np.array(prediction)
    return prediction   
                  
# Fit X_train to corresponding label in y_train
def Fit(X_train, y_train):
    X_train = np.hstack((X_train, y_train.reshape(X_train.shape[0],1)))
    return X_train
                
def main():
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
                                                        iris_dataset['target'],
                                                        random_state=0)
    
    X_train = Fit(X_train, y_train)
    result = MakePrediction(X_train, X_test, 6)
    print("Predicted: {}".format(result))
    print("Actual Class: {}".format(y_test))   
    
main()    


     
