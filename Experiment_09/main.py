import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay
digits=load_digits();X=digits.data/16.;y=digits.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=7,stratify=y)
m=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=80,random_state=7).fit(Xtr,ytr);pred=m.predict(Xte)
print("Dataset: sklearn digits (8x8), train/test:",len(Xtr),len(Xte))
print("Accuracy:",round(accuracy_score(yte,pred),4))
fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax);plt.tight_layout();plt.savefig("confusion_matrix.png")
fig,axs=plt.subplots(1,5,figsize=(9,2))
for i,a in enumerate(axs):a.imshow(Xte[i].reshape(8,8),cmap="gray");a.set_title(f"Pred {pred[i]}");a.axis("off")
plt.tight_layout();plt.savefig("sample_predictions.png")