import networkx as nx
import matplotlib.pyplot as plt
# Cloudy -> Rain -> Wet grass; Cloudy -> Sprinkler
G=nx.DiGraph([("Cloudy","Rain"),("Cloudy","Sprinkler"),("Rain","WetGrass"),("Sprinkler","WetGrass")])
p_cloud=.5;p_rain_cloud=.8;p_rain_clear=.2
p_rain=p_cloud*p_rain_cloud+(1-p_cloud)*p_rain_clear
print("P(Rain|Cloudy)=0.8; P(Rain|Clear)=0.2")
print("Marginal P(Rain):",round(p_rain,3))
pos={"Cloudy":(0,1),"Rain":(1,1),"Sprinkler":(1,0),"WetGrass":(2,.5)}
nx.draw(G,pos,with_labels=True,node_size=2500,arrows=True);plt.title("Weather Bayesian network");plt.tight_layout();plt.savefig("bayesian_network.png")