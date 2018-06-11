
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import os,sys
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster


# In[2]:


random_state = 170 # random seed

#obtencion de los nombres de los datasets
files = os.listdir("./Clustering_Datasets")

#ordenamos los nombres de los archivos
files.sort()

#for para eliminar la extercion del nombre de los archivos
for i in range(len(files)):
    files[i] = files[i][:-4]
    
#lista para guardar el resultado de la metrica ARI
result_ARI = []
for i in files:
    result_ARI.append([0.0]*4)


# In[3]:


j = 0
for i in files:
    #lectura del dataset con pandas
    dataset = pd.read_csv("./Clustering_Datasets/"+i+".txt", header=None, sep="\t")
    
    #obtenemos la dimencionalidad del dataset
    D = len(dataset.columns)-1
    
    #obtenemos el numero de clusters del dataset
    K = len(dataset[len(dataset.columns)-1].unique())
    
    #extraemos del dataset las etiquetas reales
    labels_true = dataset[len(dataset.columns)-1]
    
    #eliminamos la columna de clase del dataset
    del dataset[len(dataset.columns)-1]
    
    #normalizacion de los datos del dataset
    dataset = StandardScaler().fit_transform(dataset)
    
    #ejecusion de KMeans
    labels_kmeans = KMeans(n_clusters=K, random_state=random_state).fit_predict(dataset)

    #ejecucion de HAC averange linkage
    labels_HAC_averange = AgglomerativeClustering(n_clusters=K, linkage="average").fit_predict(dataset)
    
    #ejecucion de HAC single linkage
    links = linkage(dataset,"single")
    labels_HAC_single = fcluster(links,K,criterion="maxclust")
    
    #ejecucion de HAC complete linkage
    labels_HAC_complete = AgglomerativeClustering(n_clusters=K, linkage="complete").fit_predict(dataset)
    
    #calculo de la metrica ARI para los algoritmos de clustering
    result_ARI[j][0] = ARI(labels_true,labels_kmeans)
    result_ARI[j][1] = ARI(labels_true,labels_HAC_averange)
    result_ARI[j][2] = ARI(labels_true,labels_HAC_single)
    result_ARI[j][3] = ARI(labels_true,labels_HAC_complete)
    
    #validacion de las dimenciones del dataset para saber si se crearan los scatter plots de los algoritmos 
    if(D == 2):
        
        #creamos la figura para los scatterplots
        figure = plot.figure(figsize=(12, 12))
         
        #creamos el scatter plot del Ground Truth
        plot.subplot(321)
        plot.scatter(dataset[:, 0], dataset[:, 1], c=labels_true, edgecolors="black", linewidth=1)
        plot.title("Ground Truth",fontsize=18,fontweight="bold")
        
        #creamos el scatter plot del Kmeans
        plot.subplot(322)
        plot.scatter(dataset[:, 0], dataset[:, 1], c=labels_kmeans, edgecolors="black", linewidth=1)
        plot.title("KMeans",fontsize=18,fontweight="bold")
        
        #creamos el scatter plot del HAC Average
        plot.subplot(323)
        plot.scatter(dataset[:, 0], dataset[:, 1], c=labels_HAC_averange, edgecolors="black", linewidth=1)
        plot.title("HAC Average",fontsize=18,fontweight="bold")
        
        #creamos el scatter plot del HAC single
        plot.subplot(324)
        plot.scatter(dataset[:, 0], dataset[:, 1], c=labels_HAC_single, edgecolors="black", linewidth=1)
        plot.title("HAC Single",fontsize=18,fontweight="bold")
        
        #
        plot.subplot(325)
        plot.scatter(dataset[:, 0], dataset[:, 1], c=labels_HAC_complete, edgecolors="black", linewidth=1)
        plot.title("HAC Complete",fontsize=18,fontweight="bold")
        
        plot.suptitle(i,fontsize=18,fontweight="bold")
        plot.savefig(i)
        plot.close()
        
    j += 1


# In[ ]:



dfARI = pd.DataFrame(,columns=crimes,index=forces)


dfARI.to_csv("results_ARI.csv")

