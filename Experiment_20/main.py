import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
# Use local synthetic faces-like arrays to avoid network dataset downloads.
rng=np.random.default_rng(3);base=rng.random((64,64));imgs=np.array([np.clip(base+0.08*rng.normal(size=(64,64)),0,1) for _ in range(12)])
X=imgs.reshape(12,-1)
for k in [2,5,10]:
 p=PCA(n_components=k).fit(X);rec=p.inverse_transform(p.transform(X));err=np.mean((X-rec)**2);print("Components",k,"reconstruction MSE",round(err,5))
fig,axs=plt.subplots(1,4,figsize=(8,2))
for ax,im,title in zip(axs,[imgs[0],PCA(n_components=2).fit(X).inverse_transform(PCA(n_components=2).fit(X).transform(X)[0:1]).reshape(64,64),PCA(n_components=5).fit(X).inverse_transform(PCA(n_components=5).fit(X).transform(X)[0:1]).reshape(64,64),PCA(n_components=10).fit(X).inverse_transform(PCA(n_components=10).fit(X).transform(X)[0:1]).reshape(64,64)],["Original","2 PCs","5 PCs","10 PCs"]):ax.imshow(im,cmap="gray");ax.set_title(title);ax.axis("off")
plt.tight_layout();plt.savefig("pca_reconstruction.png")