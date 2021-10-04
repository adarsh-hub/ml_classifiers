import mltools as ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

X = np.genfromtxt('data/X_train.txt', delimiter=None)
Y = np.genfromtxt('data/Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)
Xtr, Ytr = X[:10000], Y[:10000]
Xva, Yva = X[10000:20000], Y[10000:20000]

scores = []
s_max = 0
for i in range(1,100):
    for j in range(1,100):
        hl = (i,j)
        n = MLPClassifier(solver='adam',random_state=1, hidden_layer_sizes=hl, max_iter=500).fit(Xtr, Ytr)
        s = n.score(Xva,Yva)
        scores.append([hl,s])
        if(s>s_max):
            s_max = s
            print("Hidden Layer Size:" + str(hl))
            print("Accuracy:" + str(s_max))

print(scores)
