import networkx as nx
import numpy as np
import sys
import logging
import pickle
import time
from graph_kernels import optimal_assign_kernel

np.random.seed(None)

def load_data(ds_name,use_node_labels=False):
    node2graph = {}
    Gs = []
    
    with open("./datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1
    
    with open("./datasets/%s/%s_A.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
        with open("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), "r+b") as f:
            c = 1
            for line in f:
                node_label = int(line[:-1])
                Gs[node2graph[c]-1].node[c]['label'] = node_label
                c += 1
        
    labels = []
    with open("./datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), "r") as f:
        for line in f:
            labels.append(int(line[:-1]))
    
    labels  = np.array(labels, dtype=np.float)
    return Gs, labels

def compute_kernel(graphs, kernel, method):
    start_time = time.time()	
    K = kernel(graphs, method)
    end_time = time.time()
    logger.info("Total time for kernel: %g"%((end_time - start_time)))
    return K


if __name__ == "__main__":
    # start logger
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    OUTPUT_DIR = "."

    # read the parameters
    ds_name = sys.argv[1]
    base_method = sys.argv[2]

    graphs_old, labels = load_data(ds_name, False)
    perm = np.random.permutation(len(graphs_old))
    labels = labels[perm]
    graphs=[]
    for i in range(len(graphs_old)):
        graphs.append(graphs_old[perm[i]])
    np.save(ds_name+"_labels", labels)
    print("Building kernel for "+ds_name+" with "+base_method+" kernel") 

    kernel = optimal_assign_kernel
    if base_method == "sp":
        method = "EOA-SP"
    elif base_method == "eoa":
        method = "EOA"

    K = compute_kernel(graphs, kernel, method)
    np.save(ds_name+"_"+base_method, K)

