import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
t=np.linspace(0,8,1000);s1=np.sin(2*np.pi*1*t);s2=np.sign(np.sin(2*np.pi*.5*t));S=np.c_[s1,s2];A=np.array([[1,.5],[.4,1.]]);mix=S@A.T
ica=FastICA(n_components=2,random_state=1,max_iter=1000);rec=ica.fit_transform(mix)
print("Mixing matrix:\n",A)
print("Recovered components shape:",rec.shape)
fig,axs=plt.subplots(3,1,figsize=(8,5));axs[0].plot(s1);axs[0].set_title("Source 1");axs[1].plot(s2);axs[1].set_title("Source 2");axs[2].plot(rec);axs[2].set_title("ICA separated signals");plt.tight_layout();plt.savefig("ica_signals.png")