import numpy as np
import matplotlib.pyplot as plt
x=np.array([.2,.5,.1,.8]);h=0.;Wx=.7;Wh=.5
states=[];outs=[]
for t,xt in enumerate(x):
 h=np.tanh(Wx*xt+Wh*h);states.append(h);outs.append(h)
 print(f"t={t}: input={xt:.2f}, hidden={h:.4f}, output={h:.4f}")
fig,ax=plt.subplots(figsize=(8,3));ax.plot(range(4),states,marker="o");ax.set_xticks(range(4));ax.set_xlabel("Time step");ax.set_ylabel("Hidden state");ax.set_title("Unfolded toy RNN states");plt.tight_layout();plt.savefig("rnn_states.png")