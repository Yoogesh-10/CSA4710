import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
rng=np.random.default_rng(5);t=np.arange(350);y=30+5*np.sin(t/15)+2*np.sin(t/4)+rng.normal(0,.3,len(t))
w=15;X=np.array([y[i:i+w] for i in range(len(y)-w)]);target=y[w:];s=260
configs=[("RNN",(8,)),("LSTM-inspired",(24,)),("GRU-inspired",(16,16))];scores=[]
for name,units in configs:
 m=MLPRegressor(hidden_layer_sizes=units,max_iter=350,random_state=4).fit(X[:s],target[:s]);pred=m.predict(X[s:]);mae=np.mean(abs(pred-target[s:]));scores.append(mae);print(name,"MAE",round(mae,4))
plt.bar([c[0] for c in configs],scores);plt.ylabel("MAE");plt.title("Sequence model comparison");plt.tight_layout();plt.savefig("sequence_models.png")