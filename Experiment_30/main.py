import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(5)
models={"VAE":rng.normal(.5,.12,(100,64)),"GAN":rng.normal(.5,.2,(100,64)),"Other":rng.normal(.5,.16,(100,64))}
for name,x in models.items():
 x=np.clip(x,0,1);div=np.mean(np.std(x,axis=0));quality=np.mean((x-.5)**2)
 print(name,"diversity proxy",round(div,4),"mean squared deviation",round(quality,4))
plt.boxplot([np.std(x,axis=1) for x in models.values()],labels=list(models));plt.ylabel("Per-sample pixel std");plt.title("Generated sample diversity proxy");plt.tight_layout();plt.savefig("generative_metrics.png")