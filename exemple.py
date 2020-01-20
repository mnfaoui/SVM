# -*- coding: utf-8 -*-

# Imports
import numpy as np
import math
import cPickle
from collections import Counter

# Fonction de chargement des data de cifar-10
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# Fonction de calcul de distance entre 2 éléments
def dist(el1,el2):
    dist=0.0
    dist=math.sqrt(np.sum(np.square(el1-el2)))
    return dist

# Données d'apprentissage
data_1_open=unpickle("cifar-10-batches-py/data_batch_1")
data1=data_1_open[b'data']
data1_labels=data_1_open[b'labels']

data_2_open=unpickle("cifar-10-batches-py/data_batch_2")
data2=data_2_open[b'data']
data2_labels=data_2_open[b'labels']

data_3_open=unpickle("cifar-10-batches-py/data_batch_3")
data3=data_3_open[b'data']
data3_labels=data_3_open[b'labels']

data_4_open=unpickle("cifar-10-batches-py/data_batch_4")
data4=data_4_open[b'data']
data4_labels=data_4_open[b'labels']

data_5_open=unpickle("cifar-10-batches-py/data_batch_5")
data5=data_5_open[b'data']
data5_labels=data_5_open[b'labels']

print("Longueur data1 : {} taille image = {} ".format(len(data1),len(data1[0])))
print("Longueur data2 : {} taille image = {} ".format(len(data2),len(data2[0])))
print("Longueur data3 : {} taille image = {} ".format(len(data3),len(data3[0])))
print("Longueur data4 : {} taille image = {} ".format(len(data4),len(data4[0])))
print("Longueur data5 : {} taille image = {} ".format(len(data5),len(data5[0])))

# Données de test
data_test_open=unpickle("cifar-10-batches-py/test_batch")
data_test=data_test_open[b'data']
test_labels=data_test_open[b'labels']

print("Test de reconnaissance d'image sur la base de donnees CIFAR-10")
print("Longueur data_test : {} taille image = {} ".format(len(data_test),len(data_test[0])))

# Nombre d'images à tester - 1
#a_tester=len(data_test)
a_tester=2

# Nombre de plus proches voisins à garder
K=5

# Calcul des distances et prédiction des labels
for image_number in range(0,a_tester-1):
    print("Test image num {}/{}".format(image_number+1,a_tester-1))
    # Initialisation des vecteurs contenant toutes les distances et les K minimales
    distances=[]
    distances_minimales=[]
    
    # On parcours toutes les bases de données
    for i in range(1,6):
        data_apprentissage=unpickle("cifar-10-batches-py/data_batch_"+str(i))
        
        # On calcule la distance entre l'image test et toutes les images d'une base de données
        for j in range(0,len(data_apprentissage[b'data'])):
            print("Image num{} de la section d'apprentissage {}".format(j,i))
            dist_temp=dist(data_test[image_number],data_apprentissage[b'data'][j])
            distances.append(dist_temp)
            print("Distance calculee : {}".format(distances[j]))
        #end for
    #end for
            
    # On garde les K plus petites distances
    list_dist=list(distances) # Conversion np.array -> liste
    distances_minimales=sorted(list_dist)
    
    # Initialisation des vecteurs finaux pour fixer leurs dimensions
    list_of_labels=data1_labels+data2_labels+data3_labels+data4_labels+data5_labels # Liste de tous les labels des images d'apprentissage
    distances_finales=[0,0,0,0,0] # Contient les distances min
    real_index=[0,0,0,0,0] # Contient les index de list_dist des dist min pour retrouver leur label
    label_choisi=[0,0,0,0,0] # Contient les labels choisis
        
    # Remplissage des vecteurs finaux et récupération des labels correspondants
    for k in range(0,K-1):
        real_index[k]=list_dist.index(distances_minimales[k])
        distances_finales[k]=distances_minimales[k]
        label_choisi[k]=list_of_labels[real_index[k]]
    #end for
    label_count=[0,0,0,0,0,0,0,0,0,0]
    # On prend une décision
    for n in range(0,10):
        label_count[n]=label_choisi.count(n)
    #end for
    label_final=label_count.index(max(label_count))
    print("Label choisi : {}".format(label_final))
    print("Label exact : {}".format(test_labels[image_number]))   
#end for
        











    
    
