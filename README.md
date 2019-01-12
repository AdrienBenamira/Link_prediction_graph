
In this work we address the task of link prediction in a citation network.
This work is also a part of an in-class  \href{https://www.kaggle.com/c/ngsa-w19}{Kaggle Competition for Network Course
Analytics} Course offered at Ecole CentraleSupelec, Paris in Fall 2018-2019.
Our final F-score is 0.973 on the public test set and we are currently ranked XXX / XXX.

# Set up

Put glove folder in the dataset path
Config default
``` python
run main.py
```

# Features are:

#0 overlap_title,
#1 temp_diff,
#2  comm_auth,
#3 num_inc_edges,
#4  Distance_abstract,
#5  Distance_title,
#6 shortest_path_dijkstra
#7 shortest_path_dijkstra_und
#8 ,comm_neighbors,
#9 no_edge,
#10  tfidf_distance_corpus,
#11  tfidf_distance_titles,
#12 jaccard_und
#13 Resource_allocation

# Extra :

Model_tunning.ipynb and Features.ipynb analyse our results

Todo :

adamic_adar_index
preferential_attachment


