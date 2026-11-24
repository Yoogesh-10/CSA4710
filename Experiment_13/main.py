import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
t=np.arange(240);temp=20+8*np.sin(2*np.pi*t/30)+.02*t
window=12;X=np.array([temp[i:i+window] for i in range(len(temp)-window)]);y=temp[window:]
split=int(.8*len(X));Xtr,Xte,ytr,yte=X[:split],X[split:],y[:split],y[split:]
m=MLPRegressor(hidden_layer_sizes=(32,),max_iter=80,random_state=1).fit(Xtr,ytr);p=m.predict(Xte)
print("Sequences shape:",X.shape,"Train/test:",len(Xtr),len(Xte))
print("Test MAE:",round(np.mean(abs(p-yte)),3))
plt.plot(yte,label="Actual");plt.plot(p,label="Predicted");plt.legend();plt.title("Temperature forecast");plt.tight_layout();plt.savefig("forecast.png")