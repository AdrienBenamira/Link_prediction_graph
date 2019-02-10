
# General

In this work we address the task of link prediction in a citation network.
This work is also a part of an in-class [Kaggle Competition for Network Course
Analytics](https://www.kaggle.com/c/ngsa-w19) Course offered at Ecole CentraleSupelec, Paris in Fall 2018-2019.

Our final F-score is 0.973 on the public test set and we are currently ranked 2nd / 46.

# Set up

Put glove folder in the dataset path
Config default
``` python
run main.py
```

# Features

We have :

    * overlap_title,
    * temp_diff,
    * comm_auth,
    * num_inc_edges,
    * Distance_abstract,
    * Distance_title,
    * shortest_path_dijkstra
    * shortest_path_dijkstra_und
    * comm_neighbors,
    * no_edge,
    * tfidf_distance_corpus,
    * tfidf_distance_titles,
    * jaccard_und
    * Resource_allocation



![Alt text](images/correlation.png?raw=true "Title")

![Alt text](images/importance_GB.png?raw=true "Title")

![Alt text](images/importance_RF.png?raw=true "Title")

![Alt text](images/nbre_feat.png?raw=true "Title")

# Results

Report is available [here](https://github.com/AdrienBenamira/Link_prediction_graph/blob/master/NGSA_Kaggle_report.pdf)

| Model        | Train           | Validation  |
| ------------- |:-------------:| -----:|
| Gradient Boosting | 0.979 | 0.976|
| Random Forest | 1 | 0.975 |
| SVM | 0.964 | 0.964|
| Linear | 0.966 | 0.966 |


# Extra :

Model_tunning.ipynb and Features.ipynb analyse our results

