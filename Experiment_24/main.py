import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
rng=np.random.default_rng(2);N=10000
d=rng.random(N)<.08;fever=rng.random(N)<np.where(d,.7,.1);cough=rng.random(N)<np.where(d,.6,.15)
print("Sampled disease prevalence:",round(d.mean(),3))
print("P(Disease|fever,cough):",round(d[fever&cough].mean(),3))
print("P(Disease|no symptoms):",round(d[~fever&~cough].mean(),3))
G=nx.DiGraph([("Disease","Fever"),("Disease","Cough")]);nx.draw(G,with_labels=True,node_size=2200);plt.title("Medical graphical model");plt.tight_layout();plt.savefig("medical_graph.png")