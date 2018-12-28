
The features computed are:
* The three **default features** provided with the baseline <br>
    * **Title overlap**: number of overlapping words in titles
    * **Year difference** between articles
    * **Number of common authors**
* The **TF-IDF** (Term Frequency - Inverse Document Frequency) between the source and target articles:
    * **TF-IDF distance between the articles' abstracts**
    * **TF-IDF distance between the articles' titles**
* The **Glove distance** between the source and target articles:
    * **Glove distance between the articles' abstracts**
    * **Glove distance between the articles' titles**
* Number of common neighboord
* Number of edge in common
* The **number of times the target article is cited**
* The **shortest path between the source and the target articles** (discounting an existing direct edge for the training set)
    * in the **directed** graph
    * in the **undirected** copy of the graph
* The **jaccard similarity coefficients of the source and target articles**
* The **Resource allocation index **