import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class KNN:    
    Numb_Neighbors = 1
    
    def __init__(self, Numb_Neighbors):
        self.Numb_Neighbors = Numb_Neighbors
    
    @staticmethod
    def SelectionSort(row): #each row is a data point
        for i in range(len(row)):
            minPosition = i
            min = row[i][1] # access the second subelement in the ith element of the row
            for j in range(i + 1,len(row)):
                if row[j][1] <= min:
                    min = row[j][1]
                    minPosition = j
            temp = row[i]
            row[i] = row[minPosition]
            row[minPosition] = temp
            
    @staticmethod
    def EuclideanDistance(X_train, X_test):
        distance = 0.0
        for i in range(len(X_test)):
            distance = distance + ((X_train[i] - X_test[i])**2)
        return math.sqrt(distance) 
    
    
    def get_neighbors(self, X_train, X_test):
        distances = list()
        for train_row in X_train:
            dist = KNN.EuclideanDistance(train_row, X_test)
            distances.append((train_row, dist)) 
            # the distances is a list in which each element contains 2 subelements which are an array and the distance    
            # for example: distances = (array([6.6, 2.9, 4.6, 1.3, 1. ]), 0.8602325267042624)                                                                                                          
        # print('This is distances: ',distances)
        KNN.SelectionSort(distances)    
        neighbors = list()
        for i in range(self.Numb_Neighbors):
            neighbors.append(distances[i][0])
        print('this is neighbors: ', neighbors)
        return neighbors 
    
    def MakePrediction(self, training_set, test_set):
        prediction = list()
        for test_point in test_set:
            neighbors = KNN.get_neighbors(self, training_set, test_point)
            for row in neighbors:
                output_value = [row[-1]]
            predic = max(output_value, key=output_value.count)
            prediction.append(int(predic))
            neighbors.clear()
        prediction = np.array(prediction)
        return prediction  
    
    
    def Fit(self, X_train, y_train):
        X_train = np.hstack((X_train, y_train.reshape(X_train.shape[0],1)))
        return X_train
    


def main():
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
                                                        iris_dataset['target'],
                                                        random_state=0)
    
    knnClassifier = KNN(6)
    X_train = knnClassifier.Fit(X_train, y_train)
    print(knnClassifier.MakePrediction(X_train, X_test))
    
    # distances = ((np.array([6.6, 2.9, 4.6, 1.3, 1. ]), 0.8602325267042624), (np.array([1,3,4,5,6,7]), 3))
    # print(distances[1][1])
    
    
    
   
    
    
    
main()