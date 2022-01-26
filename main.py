import csv

from layer import *

file = open('mnist_train.csv')
reader = csv.reader(file)
rows = []
for row in reader:
    row = [int(i) for i in row]
    rows.append(row)
file.close()
training_data = np.asarray(rows).reshape(60000, 785)

eta = 0.1

d1 = DenseLayer(784, 300)
a1 = ActivationLayer('sigmoid')
d2 = DenseLayer(300, 10)
a2 = ActivationLayer('sigmoid')

for i in range(0, len(training_data)):
    input = np.asarray(training_data[i][1:]).reshape(-1, 1) / 255

    label = np.zeros(10).reshape(-1, 1) 
    label[training_data[i][0]] = 1
    L = LossLayer('mse', label)

    print('i: ', i)

    #Forward Pass
    fp = d1.feedForward(input)
    fp = a1.feedForward(fp)
    fp = d2.feedForward(fp)
    fp = a2.feedForward(fp)
    fp = L.feedForward(fp)

    #Backward Pass
    bp = L.getGradient(L.input)

    bp *= a2.getGradient(L.input)
    d2.weights += np.dot(bp, d2.getGradient(d2.weights).T) * eta
    d2.bias += np.dot(bp.T, d2.getGradient(d2.bias)).T * eta
    bp = np.dot(d2.getGradient(d2.input).T, bp)

    bp *= a1.getGradient(d2.input)
    d1.weights += np.dot(bp, d1.getGradient(d1.weights).T) * eta
    d1.bias += np.dot(bp.T, d1.getGradient(d1.bias)).T * eta
    
#Test

file = open('mnist_test.csv')
reader = csv.reader(file)
rows = []
for row in reader:
    row = [int(i) for i in row]
    rows.append(row)
file.close()
test_data = np.asarray(rows).reshape(10000, 785)

counter = 0
for i in range(0, len(test_data)):
    input = np.asarray(test_data[i][1:]).reshape(-1, 1) / 255

    fp = d1.feedForward(input)
    fp = a1.feedForward(fp)
    fp = d2.feedForward(fp)
    fp = a2.feedForward(fp)
    
    print('Prediction: ', np.argmax(fp), '. True Value: ', test_data[i][0])
    if np.argmax(fp) == test_data[i][0]:
        counter += 1

print('Accuracy: ', counter/len(test_data) * 100, '%')