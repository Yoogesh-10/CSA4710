import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
rng=np.random.default_rng(2);n=200;service=rng.normal(size=n);value=rng.normal(size=n)
X=np.c_[service+rng.normal(0,.3,n),service+rng.normal(0,.3,n),value+rng.normal(0,.3,n),value+rng.normal(0,.3,n),service*.4+value*.5+rng.normal(0,.4,n)]
fa=FactorAnalysis(n_components=2,random_state=1).fit(X)
print("Factor loadings (rows=features):\n",np.round(fa.components_.T,3))
print("Interpretation: factor patterns suggest service and value dimensions.")
plt.imshow(fa.components_,aspect="auto");plt.colorbar();plt.xlabel("Latent factors");plt.ylabel("Observed features");plt.title("Factor loadings");plt.tight_layout();plt.savefig("factor_loadings.png")