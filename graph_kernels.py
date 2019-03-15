import copy
import time
from collections import defaultdict
from math import sqrt

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.cluster import hierarchy
from scipy.sparse.linalg import eigs
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import lil_matrix
from spherecluster.spherical_kmeans import SphericalKMeans
from sklearn.preprocessing import normalize

def optimal_assign_kernel(graphs, method, dim=2, K=5, L=None):
    Us = []
    index = []
    labels = []
    for i,G in enumerate(graphs):
        n = G.number_of_nodes()
        A = nx.adjacency_matrix(G).astype(float)
        if n > dim+1:
            Lambda,U = eigs(A, k=dim, ncv=10*dim)
            idx = Lambda.argsort()[::-1]
            U = U[:,idx]
        elif n == dim+1:
            Lambda,U = np.linalg.eig(A.todense())
            idx = Lambda.argsort()[::-1]   
            U = U[:,idx]
            U = U[:,:dim]
        else:
            Lambda,U_red = np.linalg.eig(A.todense())
            idx = Lambda.argsort()[::-1]   
            U_red = U_red[:,idx]
            U = np.zeros((n,dim))
            U[:,:U_red.shape[1]] = U_red
        U = np.absolute(U)
        #for j in range(U.shape[0]):
        #   U[j,:] = U[j,:]/np.linalg.norm(U[j,:])
        Us.append(U)
        assert U.shape[1] == dim, "Wrong shape of Embeddings" 
        index.append(np.ones([G.number_of_nodes(),1])*i)

    index = np.vstack(index).astype(int)
    embeddings = np.vstack(Us)
    
    if method == "EOA":
        tree, path, leaves = generate_tree(embeddings, K, L)
    else:
        tree, path, leaves = generate_tree(embeddings, K, L, 0, 0)

    num_features = tree.number_of_nodes() - len(leaves)
    hists = np.zeros([len(graphs), num_features])
    #hists = lil_matrix((len(graphs), num_features))
    
    for i in range(embeddings.shape[0]):
       for node in path[i][:-1]:
           hists[index[i], node-len(leaves)] += tree.node[node]['omega']
    
    K = np.zeros([len(graphs), len(graphs)])
    for i in range(len(graphs)):
        for j in range(i, len(graphs)):
            K[i,j] = np.sum(np.minimum(hists[i,:],hists[j,:]))#np.sum(hists[i,:].minimum(hists[j,:])) #np.sum(np.minimum(hists[i,:],hists[j,:]))
            K[j,i] = K[i,j]
    return K

def generate_tree(data, K, L, use_spherical=True, use_weights=True):
    if use_spherical:
        x,weights,_ = create_hierarchy_spherical(data,K,L)
    else:
        x,weights,_ = create_hierarchy(data,K,L)

    tree = nx.DiGraph(x)
    print(tree.number_of_nodes())
    print(tree.number_of_edges())
    outdegs = tree.out_degree()

    leaves = [n[0] for n in outdegs if outdegs[n[0]]==0]
    root = tree.number_of_nodes()-1
    path = nx.shortest_path(tree, source=root)

    if use_weights:
        for node in tree.nodes():
            if node in weights:
                tree.node[node]['weight'] = weights[node]
            else: 
                tree.node[node]['weight'] = 1

        for node in tree.nodes():
            if list(tree.predecessors(node)):
                parent = list(tree.predecessors(node))[0]        
                tree.node[node]['omega'] = tree.node[node]['weight'] - tree.node[parent]['weight'] 
                #if tree.node[node]['omega']<0:
                #tree.node[node]['omega'] == 0
            else:
                tree.node[node]['omega'] = tree.node[node]['weight']
    else:
        #for node in tree.nodes():
        #    tree.node[node]['omega'] = len(path[node])

        depth = max(np.array([len(path[node]) for node in leaves]))
        for node in tree.nodes():
            if node in leaves:
                tree.node[node]['weight'] = depth
            else:
                tree.node[node]['weight'] = len(path[node])
        
        for node in tree.nodes():
           if list(tree.predecessors(node)):
               parent = list(tree.predecessors(node))[0]        
               tree.node[node]['omega'] = tree.node[parent]['weight']/tree.node[node]['weight']
           else:
               tree.node[node]['omega'] = 1
    return tree, path, leaves

def create_hierarchy_spherical(data, K=2, L=None):
    hierarchy = {}   

    # level 0：
    index = np.arange(data.shape[0]) 
    m = np.mean(data, axis=0)
    m = m / np.linalg.norm(m)
    d = np.dot(data, m.reshape(-1,1)) 
    avg_sim = np.mean(d)
        
    hierarchy[0] = {}
    hierarchy[0][0] = [index, m, avg_sim, [], 0, [1]]

    if L:
        for i in range(1,L+2):
            print("level {}".format(i))
            hierarchy[i] = {}
            count = 0
            if i == L+1:
                for key in hierarchy[i-1]:
                    if hierarchy[i-1][key][4]==1:
                        continue
                    hierarchy[i][count] = [hierarchy[i-1][key][0], hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1, hierarchy[i-1][key][5]]
                    count+=1
            else:                    
                for key in hierarchy[i-1]:
                    idx = hierarchy[i-1][key][0]
                    if hierarchy[i-1][key][4]==1:
                        continue                    
                    if len(idx) <= K:
                        hierarchy[i][count] = [idx, hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1, hierarchy[i-1][key][5]]
                        count+=1
                        continue                        

                    points = data[idx]
                    if i>=1:
                        estim = SphericalKMeans(K)
                    else:
                        estim = SphericalKMeans(K, centers_old=hierarchy[i-1][key][1], dist_old=hierarchy[i-1][key][5])
                    estim.fit_transform(points)
                    
                    if np.max(estim.labels_)>=K:
                        idx_p = idx[estim.labels_==K]
                        dist = estim.best_dist[estim.labels_==K]
                        avg_sim = np.mean(np.dot(estim.data_[estim.labels_==K,:], estim.cluster_centers_[K,:].reshape(-1,1)))
                        hierarchy[i][count] = [idx_p, hierarchy[i-1][key][1], avg_sim, key, 1, dist]
                        count+=1                
                        
                        idx = idx[estim.labels_<K]
                        labels = estim.labels_[estim.labels_<K]
                        dists = estim.best_dist[estim.labels_<K]                      
                    else:
                        labels = estim.labels_
                        dists = estim.best_dist
                    
                    clusters = np.unique(labels)
                                        
                    if len(clusters) == 1:            
                        # hierarchy[i-1][key][4] = 1
                        # continue
                        avg_sim = np.mean(np.dot(estim.data_[estim.labels_==clusters[0],:], estim.cluster_centers_[clusters[0],:].reshape(-1,1)))          
                        dist = dists[labels==clusters[0]]
                        hierarchy[i][count] = [idx, estim.cluster_centers_[clusters[0],:], avg_sim, key, 1, dist]
                        count+=1
                        continue

                    for j in clusters:
                        idx_p = idx[labels==j]
                        dist = dists[labels==j]
                        avg_sim = np.mean(np.dot(estim.data_[estim.labels_==j,:], estim.cluster_centers_[j,:].reshape(-1,1)))      
                        hierarchy[i][count] = [idx_p, estim.cluster_centers_[j,:], avg_sim, key, 0, dist]
                        count += 1    

            if len(hierarchy[i]) == 0:
                hierarchy.pop(i)
                i -= 1
                break

            

    else:
        i = 1
        while True:           
            print("level {}".format(i))
            hierarchy[i] = {}
            count = 0
            for key in hierarchy[i-1]:
                idx = hierarchy[i-1][key][0]
                
                # check if it is a leaf
                if hierarchy[i-1][key][4]==1:
                    continue
                
                # check if it belongs to the previous level 
                if hierarchy[i-1][key][4]==2:
                   hierarchy[i][count] = [idx, hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1, hierarchy[i-1][key][5]]   
                   count+=1
                   continue   

                # check if it has more nodes than cluster number 
                if len(idx) <= K:            
                    # hierarchy[i-1][key][4] = 1
                    # continue
                    hierarchy[i][count] = [idx, hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1, hierarchy[i-1][key][5]]
                    count+=1
                    continue
                
                points = data[idx]
                if i == 1:
                    estim = SphericalKMeans(K)
                else:
                    estim = SphericalKMeans(K, centers_old=hierarchy[i-1][key][1], dist_old=hierarchy[i-1][key][5])
                
                estim.fit_transform(points)

                # check if there are nodes belong to previous level                                
                if np.max(estim.labels_)>=K:
                   idx_p = idx[estim.labels_==K]
                   avg_sim = np.mean(np.dot(estim.data_[estim.labels_==K,:], estim.cluster_centers_[K,:].reshape(-1,1)))
                   dist = estim.best_dist[estim.labels_==K]
                   hierarchy[i][count] = [idx_p, hierarchy[i-1][key][1], avg_sim, key, 1, dist]
                   count+=1                
                   
                   idx = idx[estim.labels_<K]
                   labels = estim.labels_[estim.labels_<K]
                   dists = estim.best_dist[estim.labels_<K]
                else:
                   labels = estim.labels_
                   dists = estim.best_dist

                clusters = np.unique(labels)
                
                if len(clusters) == 1:  
                    # hierarchy[i-1][key][4] = 1
                    # continue
                    avg_sim = np.mean(np.dot(estim.data_[estim.labels_==clusters[0],:], estim.cluster_centers_[clusters[0],:].reshape(-1,1)))          
                    dist = dists[labels==clusters[0]]
                    hierarchy[i][count] = [idx, estim.cluster_centers_[clusters[0],:], avg_sim, key, 1, dist]
                    count+=1
                    continue

                for j in clusters:
                    idx_p = idx[labels==j]
                    avg_sim = np.mean(np.dot(estim.data_[estim.labels_==j,:], estim.cluster_centers_[j,:].reshape(-1,1)))  
                    dist = dists[labels==j]    
                    hierarchy[i][count] = [idx_p, estim.cluster_centers_[j,:], avg_sim, key, 0, dist]
                    count += 1            
            
            if len(hierarchy[i]) == 0:
                hierarchy.pop(i)
                i -= 1
                break
            else:
                i += 1

    num_layer = copy.copy(i)
    adjacency_list = {}
    weight_list = {}
    center_list = {}
    count = data.shape[0] 
    for i in range(0, num_layer+1)[::-1]:
        for key in hierarchy[i]:
            if hierarchy[i][key][4]==1: #len(hierarchy[i][key][0])<=K or : 
                adjacency_list[count+key] = list(hierarchy[i][key][0])            
            weight_list[count+key] = hierarchy[i][key][2]
            center_list[count+key] = hierarchy[i][key][1]
            if i > 0:
                parent = count+len(hierarchy[i])+hierarchy[i][key][3]
                if parent in adjacency_list:
                    adjacency_list[parent].append(count+key)
                else:
                    adjacency_list[parent] = [count+key]
        count += len(hierarchy[i])
    
    return adjacency_list, weight_list, center_list

def create_hierarchy(data, K=2, L=None):
    hierarchy = {}
    index = np.arange(data.shape[0])    

    index = np.arange(data.shape[0]) 
    m = np.mean(data, axis=0)
    dist = pairwise_distances(data, m.reshape(1,-1))
    d = max(dist) 
            
    hierarchy[0] = {}
    hierarchy[0][0] = [index, m, d, [], 0]
    
    if L:
        for i in range(1,L+2):
            print("level {}".format(i))
            hierarchy[i] = {}
            count = 0
            if i == L+1:
                for key in hierarchy[i-1]:
                    if hierarchy[i-1][key][4]==1:
                        hierarchy[i][count] = hierarchy[i-1][key]
                        count+=1
                        continue
                    hierarchy[i][count] = [hierarchy[i-1][key][0], hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1]
                    count+=1
            else:                    
                for key in hierarchy[i-1]:
                    idx = hierarchy[i-1][key][0]
                    
                    if hierarchy[i-1][key][4]==1:
                        hierarchy[i][count] = hierarchy[i-1][key]
                        count+=1                        
                        continue                    
                    
                    if len(idx) <= K:
                        hierarchy[i][count] = [idx, hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1]
                        count+=1
                        continue                        
                    
                    points = data[idx]
                    estim = KMeans(K)
                    dist = estim.fit_transform(points)
                    clusters = np.unique(estim.labels_)
                    
                    if len(clusters) == 1:            
                        hierarchy[i][count] = [idx, hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1]
                        count+=1
                        continue
                    
                    for j in clusters:
                        idx_p = idx[estim.labels_==j]
                        dist_p = dist[estim.labels_==j, j]
                        hierarchy[i][count] = [idx_p, estim.cluster_centers_[j,:], max(dist_p), key, 0]
                        count += 1
            
            #if len(hierarchy[i]) == 0:
            #    hierarchy.pop(i)
            #    i -= 1
            #    break
            flag = 1
            for key in hierarchy[i]:
                if hierarchy[i][key][4] == 0:
                    flag = 0
                    break
            if flag:
                break            
        
    else:
        i = 1
        while True:
            print("level {}".format(i))
            hierarchy[i] = {}
            count = 0
            for key in hierarchy[i-1]:
                idx = hierarchy[i-1][key][0]
                #if hierarchy[i-1][key][4]==1:
                #    continue
                if len(idx) <= K:            
                    hierarchy[i][count] = [idx, hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1]
                    count+=1
                    continue
                points = data[idx]
                estim = KMeans(K)
                dist = estim.fit_transform(points)
                clusters = np.unique(estim.labels_)
                if len(clusters) == 1:            
                    hierarchy[i][count] = [idx, hierarchy[i-1][key][1], hierarchy[i-1][key][2], key, 1]
                    count+=1
                    continue
                for j in clusters:
                    idx_p = idx[estim.labels_==j]
                    dist_p = dist[estim.labels_==j, j]
                    hierarchy[i][count] = [idx_p, estim.cluster_centers_[j,:], max(dist_p), key, 0]
                    count += 1            
            
            # unbalanced hierarchy
            #if len(hierarchy[i]) == 0:
            #    hierarchy.pop(i)
            #    i -= 1
            #    break
            #else:
            #    i += 1

            # balanced hierarchy
            flag = 1
            for key in hierarchy[i]:
                if hierarchy[i][key][4] == 0:
                    flag = 0
                    break
            if flag:
                break
            else:
                i += 1

    num_layer = copy.copy(i)
    adjacency_list = {}
    weight_list = {}
    center_list = {}
    count = data.shape[0] 
    for i in range(0, num_layer+1)[::-1]:
        for key in hierarchy[i]:
            if i == num_layer: #hierarchy[i][key][4]==1: #len(hierarchy[i][key][0])<=K or : 
                adjacency_list[count+key] = list(hierarchy[i][key][0])            
            weight_list[count+key] = hierarchy[i][key][2]
            center_list[count+key] = hierarchy[i][key][1]
            if i > 0:
                parent = count+len(hierarchy[i])+hierarchy[i][key][3]
                if parent in adjacency_list:
                    adjacency_list[parent].append(count+key)
                else:
                    adjacency_list[parent] = [count+key]
        count += len(hierarchy[i])
    return adjacency_list, weight_list, center_list