import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X,y=make_classification(n_samples=500,n_features=12,n_informative=5,random_state=4)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=4)
configs=[("Baseline",0,False),("L2",.01,False),("Early stopping",.001,True)]
scores=[]
for name,alpha,early in configs:
 m=MLPClassifier(hidden_layer_sizes=(40,20),alpha=alpha,early_stopping=early,max_iter=60,random_state=2).fit(Xtr,ytr)
 scores.append((accuracy_score(ytr,m.predict(Xtr)),accuracy_score(yte,m.predict(Xte))))
 print(name,"train/test accuracy:",np.round(scores[-1],3))
plt.bar([c[0] for c in configs],[s[1] for s in scores]);plt.ylabel("Test accuracy");plt.title("Regularization comparison");plt.tight_layout();plt.savefig("regularization.png")