import numpy as np
import matplotlib.pyplot as plt
patterns=np.array([[1,1,0,0,1,1],[1,0,0,1,1,0],[0,1,1,0,0,1]],float)
W1=np.array([[.5,.2],[.2,.5],[.4,.1],[.1,.4],[.5,.2],[.2,.5]])
H=patterns@W1;recon=(H@W1.T>1.0).astype(int)
energy=-np.sum(patterns*recon,axis=1)
print("Toy visible patterns:\n",patterns)
print("Hidden activations:\n",H)
print("Reconstructed patterns:\n",recon)
print("Toy energy values:",energy)
plt.imshow(np.vstack([patterns,recon]),aspect="auto");plt.title("Patterns (top) and reconstructions");plt.tight_layout();plt.savefig("dbm_patterns.png")