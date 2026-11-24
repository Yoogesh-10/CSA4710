import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
d=load_digits();X=d.data/16.;y=d.target;Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=1,stratify=y)
pipe=Pipeline([("rbm",BernoulliRBM(n_components=64,n_iter=5,random_state=1)),("clf",LogisticRegression(max_iter=80))]).fit(Xt,yt)
pred=pipe.predict(Xv);print("RBM-feature logistic accuracy:",round(accuracy_score(yv,pred),4))
H=pipe.named_steps["rbm"].transform(Xt[:5]);print("Learned representation shape:",H.shape)
plt.imshow(H,aspect="auto");plt.title("RBM hidden representations");plt.tight_layout();plt.savefig("dbn_features.png")