import csv
import random

class Perceptron:
    def __init__(self, number_of_weights, classes):
        self.number_of_weights = number_of_weights
        self.weights = self.generate_random_weights(number_of_weights)
        self.dict_classes = { classes[0]:1, classes[1]:-1 }

    def generate_random_weights(self, n):
        weights = []
        for x in range(n):
            weights.append(random.random()*10-5)
        print('Initial weights: {}'.format(weights))
        return weights

    def train(self, data, epochs):
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            for datum in data:
                real_class = self.dict_classes[datum[len(datum)-1]]
                prediction_class = self.predict(datum)
                #print('y:{}, prediction_class:{}'.format(y,prediction_class))
                if real_class != prediction_class:
                    self.adjust_weights(real_class,datum)
                    print('Adjusted weights: {}'.format(self.weights))
            print('Final weights from epoch {}: {}'.format(epoch,self.weights))

    def predict(self, datum):
        weights_without_bias = self.weights[1:self.number_of_weights]
        attribute_values = datum[:self.number_of_weights-1]
        weight_bias = self.weights[0]
        activation = sum([i*j for i,j in zip(weights_without_bias,attribute_values)]) + weight_bias
        return 1 if activation > 0 else -1

    def adjust_weights(self, real_class, datum):
        self.weights[0] = self.weights[0] + real_class
        for i in range(1,self.number_of_weights):
            self.weights[i] = self.weights[i] + real_class * datum[i-1]

    def verify(self, data):
        count = 0
        for datum in data:
            real_class = self.dict_classes[datum[len(datum)-1]]
            prediction_class = self.predict(datum)
            if real_class != prediction_class:
                count = count + 1
        return (1-count/len(data))*100

def read_csv(file):
    with open(file,'r') as f:
        reader = csv.reader(f)
        info = []
        for row in reader:
             info.append(row)

        for datum in info:
            for i in range(len(info[0])-1):
                datum[i] = float(datum[i])
        return info


def main():
    #Get data
    data = read_csv('flowers_dataset.csv')
    data_iris_setosa = data[0:50]
    data_iris_versicolor = data[50:100]
    data_iris_virginica = data[100:150]

    data_with_two_species = data_iris_versicolor + data_iris_virginica
    classes = ['Iris-versicolor', 'Iris-virginica']

    #Order data in random way
    random.shuffle(data_with_two_species)

    #Training and verification data
    data_training = data_with_two_species[0:90]
    data_verification = data_with_two_species[:-10]

    #Perceptron
    perceptron = Perceptron(len(data[0]), classes)

    #Train
    perceptron.train(data_training, epochs=1000)
    print('\nFinal weights from training: {}'.format(perceptron.weights))

    #Verify
    accuracy = perceptron.verify(data_verification)
    print('Error: {} %'.format(100-accuracy))



if __name__ == "__main__":
    main()
