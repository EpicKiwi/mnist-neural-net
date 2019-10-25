import mnist_loader
from network import Network

network_structure = [784, 100, 10]

print("Entrainement d'un réseau sur les données MNIST")
print("Structure du réseau : ", network_structure)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network(network_structure)

net.train_big_set(training_data, 30, 10, 3.0, test_data=test_data)