import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
rng=np.random.default_rng(1);t=np.arange(300);series=100+0.05*t+2*np.sin(t/8)+rng.normal(0,.3,300)
w=20;X=np.array([series[i:i+w] for i in range(len(series)-w)]);y=series[w:];s=220
a=MLPRegressor(hidden_layer_sizes=(16,),max_iter=60,random_state=1).fit(X[:s],y[:s])
b=MLPRegressor(hidden_layer_sizes=(32,16),max_iter=60,random_state=1).fit(X[:s],y[:s])
pa=a.predict(X[s:]);pb=b.predict(X[s:])
print("Simple model MAE:",round(np.mean(abs(pa-y[s:])),4))
print("Gated-inspired larger model MAE:",round(np.mean(abs(pb-y[s:])),4))
plt.plot(y[s:],label="Actual");plt.plot(pa,label="Simple");plt.plot(pb,label="Larger");plt.legend();plt.title("Sequence model comparison");plt.tight_layout();plt.savefig("rnn_lstm_comparison.png")