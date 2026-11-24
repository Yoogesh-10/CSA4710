import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
d=load_digits();X=d.data/16;y=d.target
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=3,stratify=y)
configs=[("Small",(32,)),("Medium",(64,32)),("Deep",(64,32,16))];acc=[]
for name,layers in configs:
 m=MLPClassifier(hidden_layer_sizes=layers,max_iter=60,random_state=3).fit(Xt,yt);s=accuracy_score(yv,m.predict(Xv));acc.append(s);print(name,layers,"accuracy",round(s,4))
plt.bar([c[0] for c in configs],acc);plt.ylabel("Accuracy");plt.title("Architecture comparison");plt.tight_layout();plt.savefig("architecture_accuracy.png")