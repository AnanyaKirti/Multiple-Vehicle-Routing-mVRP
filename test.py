
import networkx as nx
import matplotlib.pyplot as plt

G = nx.complete_graph(10)
print G.edges()
# nx.draw(G, arrows=True,  with_labels=True) 
# plt.show()
