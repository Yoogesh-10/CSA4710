import numpy as np
import matplotlib.pyplot as plt
g=np.array([.2,.5,1.2,3.,8.,15.,30.]);clipped=np.clip(g,-5,5)
print("Original gradients:",g)
print("Clipped gradients:",clipped)
print("Original norm:",round(np.linalg.norm(g),3),"Clipped norm:",round(np.linalg.norm(clipped),3))
plt.plot(g,label="Original");plt.plot(clipped,label="Clipped");plt.legend();plt.title("Gradient clipping");plt.tight_layout();plt.savefig("gradient_clipping.png")