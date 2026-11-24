import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(1);v0=np.array([1.,0.,1.,0.]);W=rng.normal(0,.4,(4,2));vb=np.zeros(4);hb=np.zeros(2)
def sig(x):return 1/(1+np.exp(-x))
v=v0.copy();errors=[]
for k in range(1,6):
 h=sig(v@W+hb);v=sig(h@W.T+vb);err=np.mean((v0-v)**2);errors.append(err);print(f"CD-{k} sample reconstruction MSE={err:.4f}")
plt.plot(range(1,6),errors,marker="o");plt.xlabel("Sampling steps");plt.ylabel("Reconstruction MSE");plt.title("CD sampling illustration");plt.tight_layout();plt.savefig("cd_errors.png")