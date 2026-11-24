import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(1)
vae=np.clip(rng.normal(.5,.15,(16,8,8)),0,1)
gan=np.clip(rng.normal(.5,.22,(16,8,8)),0,1)
print("VAE sample mean/std:",round(vae.mean(),3),round(vae.std(),3))
print("GAN sample mean/std:",round(gan.mean(),3),round(gan.std(),3))
fig,axs=plt.subplots(2,8,figsize=(8,2.5))
for row,data in enumerate([vae,gan]):
 for ax,im in zip(axs[row],data):ax.imshow(im,cmap="gray",vmin=0,vmax=1);ax.axis("off")
axs[0,0].set_ylabel("VAE");axs[1,0].set_ylabel("GAN");plt.tight_layout();plt.savefig("vae_gan_samples.png")