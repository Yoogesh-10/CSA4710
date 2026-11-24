import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(0);X=np.array([[0,0],[0,1],[1,0],[1,1]],float);y=np.array([[0],[1],[1],[0]])
W1=rng.normal(size=(2,4));b1=np.zeros((1,4));W2=rng.normal(size=(4,1));b2=np.zeros((1,1))
def sig(z):return 1/(1+np.exp(-z))
losses=[]
for epoch in range(1200):
 h=sig(X@W1+b1);o=sig(h@W2+b2);loss=-np.mean(y*np.log(o+1e-9)+(1-y)*np.log(1-o+1e-9))
 d2=(o-y)/len(y);gW2=h.T@d2;gb2=d2.sum(axis=0,keepdims=True);d1=(d2@W2.T)*h*(1-h);gW1=X.T@d1;gb1=d1.sum(axis=0,keepdims=True)
 for p,g in [(W1,gW1),(b1,gb1),(W2,gW2),(b2,gb2)]:p-=.5*g
 if epoch%500==0:losses.append(loss)
print("Loss before (initial approx):",round(losses[0],4),"after:",round(loss,4))
print("Predictions:",(o>.5).astype(int).ravel().tolist(),"Targets:",y.ravel().tolist())
plt.plot(losses);plt.title("Backprop training loss");plt.xlabel("Checkpoint");plt.ylabel("Loss");plt.tight_layout();plt.savefig("training_loss.png")