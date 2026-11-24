import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# Digits image dataset used as a compact image-classification stand-in.
d=load_digits();X=d.images/16.;y=(d.target%2)
Xtr,Xte,ytr,yte=train_test_split(X.reshape(-1,64),y,test_size=.25,random_state=1,stratify=y)
model=MLPClassifier(hidden_layer_sizes=(32,),max_iter=200,random_state=1).fit(Xtr,ytr)
pred=model.predict(Xte);print("Binary image labels: 0=healthy-like, 1=disease-like (synthetic mapping)")
print("Test accuracy:",round(accuracy_score(yte,pred),4))
fig,axs=plt.subplots(1,5,figsize=(8,2))
for i,a in enumerate(axs):a.imshow(Xte[i].reshape(8,8),cmap="gray");a.set_title(f"Pred {pred[i]}");a.axis("off")
plt.tight_layout();plt.savefig("image_predictions.png")