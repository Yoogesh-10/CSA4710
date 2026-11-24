import numpy as np
import matplotlib.pyplot as plt
X=np.array([[1,0],[0,1],[1,1],[0,0],[2,1],[0,2]],float);y=np.array([1,0,1,0,1,0])
w=np.zeros(2);b=0
for epoch in range(20):
 for xi,yi in zip(X,y):
  pred=int(xi@w+b>=0);err=yi-pred;w+=err*xi;b+=err
pred=(X@w+b>=0).astype(int)
print("Weights:",w,"Bias:",b)
print("Predictions:",pred.tolist())
print("Actual:",y.tolist(),"Accuracy:",(pred==y).mean())
plt.scatter(X[:,0],X[:,1],c=y);plt.title("Email features (class 0/1)");plt.xlabel("Suspicious words");plt.ylabel("Links");plt.tight_layout();plt.savefig("perceptron_data.png")