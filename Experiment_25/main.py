import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
X=np.array([[1,1,0,0],[1,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,0,1]],float)
rbm=BernoulliRBM(n_components=3,learning_rate=.05,n_iter=20,random_state=1).fit(X)
H=rbm.transform(X);R=1/(1+np.exp(-(H@rbm.components_+rbm.intercept_visible_)))
print("Visible data:\n",X)
print("Hidden activations:\n",np.round(H,2))
print("Reconstruction probabilities:\n",np.round(R,2))
print("Reconstruction MSE:",round(np.mean((X-R)**2),4))
plt.imshow(np.vstack([X,R]),aspect="auto",vmin=0,vmax=1);plt.title("Original rows then reconstruction");plt.tight_layout();plt.savefig("rbm_reconstruction.png")