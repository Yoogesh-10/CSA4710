import numpy as np
import matplotlib.pyplot as plt
Wh=.5;steps=np.arange(1,21);grads=Wh**steps
print("Recurrent weight:",Wh)
print("Gradient magnitudes:",np.round(grads,6))
print("Gradient at step 20:",grads[-1])
plt.semilogy(steps,grads,marker="o");plt.xlabel("Backpropagation steps");plt.ylabel("Gradient magnitude (log)");plt.title("Vanishing gradient illustration");plt.tight_layout();plt.savefig("gradient_flow.png")