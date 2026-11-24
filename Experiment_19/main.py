import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
rng=np.random.default_rng(4);attendance=rng.normal(80,8,100);internal=attendance*.3+rng.normal(60,8,100);assign=internal*.5+rng.normal(70,5,100);exam=internal*.6+assign*.3+rng.normal(10,8,100)
X=np.c_[attendance,internal,assign,exam];Z=StandardScaler().fit_transform(X);pca=PCA().fit(Z);R=pca.transform(Z)[:,:2]
print("Eigenvalues:",np.round(pca.explained_variance_,3))
print("Explained variance ratio:",np.round(pca.explained_variance_ratio_,3))
print("Cumulative:",np.round(np.cumsum(pca.explained_variance_ratio_),3))
plt.plot(range(1,5),np.cumsum(pca.explained_variance_ratio_),marker="o");plt.axhline(.9,ls="--");plt.xlabel("Components");plt.ylabel("Cumulative variance");plt.title("PCA explained variance");plt.tight_layout();plt.savefig("pca_variance.png")
plt.scatter(R[:,0],R[:,1]);plt.title("Student data in PC space");plt.tight_layout();plt.savefig("pca_scatter.png")