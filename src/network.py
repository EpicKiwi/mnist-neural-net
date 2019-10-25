import numpy as np
from functions import *
import random


class Network:
	"""
	Classe représentant le réseau de neurones dans son ensemble et toutes les fonctionnalités
	de prédictions et d'apprentissage
	"""

	def __init__(self, layers_sizes):
		self.num_layers = len(layers_sizes)  # Le nombre de couches de notre réseau
		self.layers_sizes = layers_sizes  # Le nombre de neurones dans chaque couche
		# Les biais pour chacun des neurones de chacune de couches (sauf la première)
		self.biases = [np.random.randn(y, 1) for y in layers_sizes[1:]]
		# Les poids qui lient tout les neurones précédent au neurones de la couche
		self.weights = [np.random.randn(y, x)  # X le nombre de neurones de la couche précedente, Y de l'actuel
		                for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]

	# La fonction zip associe chaque élément des deux tableau en tuples (les éléments excedentaires sont retirés)

	def feedforward(self, input):
		"""
		Calcul du résultat de chaque couche avec "input" en entrée
		input doit être une matrice de 1xn elements (en 2 dimentions)
		"""

		a = input

		if np.array(a).ndim == 1:
			a = list(map(lambda x: [x], a))
		# Dans le cas ou un simple tableau a une dimention est donné on l'ajuste

		for b, w in zip(self.biases, self.weights):
			# Calcul du produit scalaire des entrées précédentes par les poids des neurones
			weighted = np.dot(w, a)
			# Ajout des biais de chaque neurone
			biaised = weighted + b
			# Calcul de la fonction sigmoid de chaque neurone
			a = sigmoid(biaised)

		return a

	def train_big_set(self, training_data, epochs, subset_size, learning_rate, test_data=None):
		"""
		Fonction d'apprentissage du réseau par gradient decent stocastique sur les données
		de training_data en les découpants en petits subsets aléatoires et en répétant
		le processus d'aprentissages autant de fois que epochs le définis

		training_data est un liste de tuples (x,y) avec x les données d'entrée et y le résultat attendu
		epochs est le nombre de fois qu'il faut entrainer les différents subsets de training_data
		subset_size est la taille du subset sur lequel entrainer le réseau
		learning_rate est le coeficient d'apprentissage à utiliser
		:param test_data:
		"""

		training_data = list(training_data)

		# Le nombre de données d'entrainement donné
		training_data_size = len(training_data)

		n_test = 0
		if test_data is not None:
			test_data = list(test_data)
			n_test = len(test_data)

		# Pour chaque entrainement demandé
		for j in range(epochs):

			# On mélange les données de base
			random.shuffle(training_data)

			# On crée un ensemble de subsets des données sur la base des données mélangées
			subset_list = [training_data[k:k + subset_size]
			                for k in range(0, training_data_size, subset_size)]

			for subset in subset_list:
				self.train(subset, learning_rate)

			if test_data is not None:
				test_results = self.evaluate(test_data)
				print("Passe {}/{} terminée (Evaluation {}/{} {}%)"
				      .format(
						j+1,
						epochs,
						test_results,
						n_test,
						str(round((test_results/n_test)*1000)/10)))
			else:
				print("Passe {}/{} terminée".format(j + 1, epochs))

	def train(self, training_data, learning_rate):
		"""
		Fonction d'apprentissage du réseau par gradient decent sur les données contenues dans training_data

		training_data est un liste de tuples (x,y) avec x les données d'entrée et y le résultat attendu
		learning_rate est le coeficient d'apprentissage à utiliser

		Préfèrer ``train_big_set`` pour entrainer avec l'algorithme stochastique
		"""

		# L'aprentissage implique de minimiser la fonction de cout qui s'applique sur chacun des éléments
		# de training_data, la fonction de coup est en fait une moyenne des resultats obtenus sur chacun
		# des elements de traning data. Nous devon alors éfféctuer une moyenne sur les radients obtenus

		# Somme des radients obtenus par backpropagation sur les biais (nabla_b) et poids (nabla_w)
		# Initialisés à 0 pour chaque couche au début
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for input_data, result_expected in training_data:
			# Calcul des gradiants des bias et des poids pour ce set de données/resultat
			delta_nabla_b, delta_nabla_w = self.backprop(input_data, result_expected)

			# On éfféctue la somme avec les résultats des précédents sets d'entrainement
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		# On calcul les nouveaux poids et biais en calculant la moyenne des gradiants et en les divisants
		# par le learning rate
		self.weights = [w - (learning_rate / len(training_data)) * nw
		                for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (learning_rate / len(training_data)) * nb
		               for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""Renvoie le tuple de gradiants ``(nabla_b, nabla_w)`` pour les données x et le resultat y.
		``nabla_b`` la différence entre les biais actuels et ce qu'ils devraient être
        ``nabla_w`` la différence entre les poids actuels et ce qu'ils devraient être

        x sont les données input
        y est le resultat attendu pour ces données
        """

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x]  # list to store all the activations, layer by layer
		zs = []  # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
		        sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
		return (nabla_b, nabla_w)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
		return (output_activations - y)

	def evaluate(self, test_data):
		"""
		Renvoie le nombre de tests que le réseau a réussi a évaluer correctement
		"""

		# créer un tableau avec l'index du neurone qui es les plus fort en sortie
		# avec le résulta attendu
		test_results = [(np.argmax(self.feedforward(x)), y)
		                for (x, y) in test_data]
		# Renvoie le nombre de résultats justes
		return sum(int(x == y) for (x, y) in test_results)
