import random
import numpy as np
import csv
import nltk
import scipy
import igraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tqdm
import os
"""
nltk.download('punkt') # for tokenization
nltk.download('stopwords')"""



class features_dataset:

    def __init__(self, prepocess_all, ratio=0.10):


        # do some nltk stuff (for stopwords, tokenization,...)
        # nltk.download('punkt')  # for tokenization
        # nltk.download('stopwords')
        self.stpwds = set(nltk.corpus.stopwords.words("english"))
        self.stemmer = nltk.stem.PorterStemmer()

        # read training and test set
        print(os.path.dirname(__file__))
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'testing_set.txt'), "r") as f:
            reader = csv.reader(f)
            testing_set = list(reader)
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_set.txt'), "r") as f:
            reader = csv.reader(f)
            training_set = list(reader)
        self.training_set = [element[0].split(" ") for element in training_set]
        self.testing_set = [element[0].split(" ") for element in testing_set]
        to_keep = random.sample(range(len(self.training_set)), k=int(round(len(training_set)*ratio)))
        self.training_set_reduced = [self.training_set[i] for i in to_keep]

        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'node_information.csv'), "r") as f:
            reader = csv.reader(f)
            self.node_info = list(reader)
        # EXTRACT INFORMATION FROM "node_information.csv"
        # the columns of the data frame below are:
        # (1) paper unique ID (integer)
        # (2) publication year (integer)
        # (3) paper title (string)
        # (4) authors (strings separated by ,)
        # (5) name of journal (optional) (string)
        # (6) abstract (string) - lowercased, free of punctuation except intra-word dashes
        self.IDs = [element[0] for element in self.node_info]
        self.publication_years = [element[1] for element in self.node_info]
        self.titles = [element[2] for element in self.node_info]
        self.authors = [element[3] for element in self.node_info]
        self.journals = [element[4] for element in self.node_info]
        self.corpus = [element[5] for element in self.node_info]

        self.vectorizer_corpus = TfidfVectorizer(stop_words="english")
        self.features_TFIDF_corpus = self.vectorizer_corpus.fit_transform(self.corpus)

        self.vectorizer_titles = TfidfVectorizer()
        self.features_TFIDF_titles = self.vectorizer_titles.fit_transform(self.titles)

        self.vectorizer_authors = TfidfVectorizer()
        self.features_TFIDF_authors = self.vectorizer_authors.fit_transform(self.authors)

        self.features = dict()
        self.training_labels = None
        self.train_labels = None
        self.valid_labels = None
        self.prediction = None
        self.fscore_v = None
        self.fscore_t = None
        self.G = igraph.Graph(directed=True)
        self.G_und = None


    def prepocess_data(self):

        edges = [(element[0], element[1]) for element in self.training_set if element[2] == "1"]
        nodes = self.IDs
        self.G.add_vertices(nodes)
        self.G.add_edges(edges)
        self.G_und = self.G.as_undirected()

        datasets = [self.training_set_reduced, self.testing_set]
        datasets_name = ["training", "testing"]

        for k in range(2):
        # for k, type_data in enumerate([self.training_set_reduced, self.testing_set]):

            dataset = datasets[k]

            # number of overlapping words in title
            overlap_title = []
            # temporal distance between the papers
            temp_diff = []
            # number of common authors
            comm_auth = []
            # number
            num_inc_edges =[]
            #distance
            Distance_title = []

            Distance_abstract = []

            comm_neighbors = []

            no_edge = []

            tfidf_distance_corpus = []

            tfidf_distance_titles = []

            shortest_path_dijkstra = []

            shortest_path_dijkstra_und = []

            jaccard_und = []

            Resource_allocation = []

            counter = 0
            for i in tqdm.tqdm(range(len(dataset))):

                source = dataset[i][0]
                target = dataset[i][1]

                source_info = [element for element in self.node_info if element[0] == source][0]
                target_info = [element for element in self.node_info if element[0] == target][0]

                index_source = self.IDs.index(source)
                index_target = self.IDs.index(target)

                list_source = self.G.neighbors(source)
                list_target = self.G.neighbors(target)

                # convert to lowercase and tokenize
                source_title = source_info[2].lower().split(" ")
                # remove stopwords
                source_title = [token for token in source_title if token not in self.stpwds]
                source_title = [self.stemmer.stem(token) for token in source_title]

                target_title = target_info[2].lower().split(" ")
                target_title = [token for token in target_title if token not in self.stpwds]
                target_title = [self.stemmer.stem(token) for token in target_title]



                source_title_glove = self.get_glove_matrix(source_info[2].lower().split(" "))
                target_title_glove = self.get_glove_matrix(target_info[2].lower().split(" "))
                source_abstract_glove = self.get_glove_matrix(source_info[5].lower().split(" "))
                target_abstract_glove = self.get_glove_matrix(target_info[5].lower().split(" "))
                distance_title = scipy.spatial.distance.euclidean(source_title_glove, target_title_glove)
                distance_abstract = scipy.spatial.distance.euclidean(source_abstract_glove, target_abstract_glove)
                Distance_abstract.append(distance_abstract)
                Distance_title.append(distance_title)


                source_auth = source_info[3].split(",")
                target_auth = target_info[3].split(",")
                overlap_title.append(len(set(source_title).intersection(set(target_title))))
                temp_diff.append(int(source_info[1]) - int(target_info[1]))
                comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
                num_inc_edges.append(len([element[1] for element in self.training_set if element[1] == target])
                                     + len([element[1] for element in self.testing_set if element[1] == target]))

                comm_neighbors.append(len(list(set(list_source).intersection(list_target))))
                no_edge.append(self.G.edge_disjoint_paths(index_source, index_target))
                if k == 0 and dataset[i][2] == "1":
                    self.G.delete_edges((index_source, index_target))
                    self.G_und.delete_edges((index_source, index_target))
                short_path = min(100000, self.G.shortest_paths_dijkstra(source=source, target=target)[0][0])
                shortest_path_dijkstra.append(short_path)
                short_path_und = min(100000, self.G_und.shortest_paths_dijkstra(source=source, target=target)[0][0])
                shortest_path_dijkstra_und.append(short_path_und)
                if k == 0 and dataset[i][2] == "1":
                    self.G.add_edge(index_source, index_target)
                    self.G_und.add_edge(index_source, index_target)

                tfidf_distance_corpus.append(
                    linear_kernel(self.features_TFIDF_corpus[index_source:index_source + 1],
                                  self.features_TFIDF_corpus[index_target:index_target + 1]).flatten())
                tfidf_distance_titles.append(
                    linear_kernel(self.features_TFIDF_titles[index_source:index_source + 1],
                                  self.features_TFIDF_titles[index_target:index_target + 1]).flatten())

                if short_path_und > 2:
                    jacc = 0
                else:
                    jacc = self.G_und.similarity_jaccard(pairs=[(index_source, index_target)])[0]
                jaccard_und.append(jacc)

                Resource_allocation.append(
                    sum(1 / self.G_und.degree(w) for w in list(set(list_source).intersection(list_target))))

                counter += 1
                if counter % 1000 == True:
                    print(counter, "training examples processsed")



            self.training_labels = np.array([int(element[2]) for element in self.training_set_reduced])

            if type_data == self.training_set_reduced:
                np.save('./features_data/shortest_path_dijkstra_train.npy', np.array(shortest_path_dijkstra))
                np.save('./features_data/shortest_path_dijkstra_und_train.npy', np.array(shortest_path_dijkstra_und))
                np.save('./features_data/comm_neighbors_train.npy', np.array(comm_neighbors))
                np.save('./features_data/no_edge_train.npy', np.array(no_edge))
                np.save('./features_data/overlap_title_train.npy', np.array(overlap_title))
                np.save('./features_data/temp_diff_train.npy', np.array(temp_diff))
                np.save('./features_data/comm_auth_train.npy', np.array(comm_auth))
                np.save('./features_data/num_inc_edges_train.npy', np.array(num_inc_edges))
                np.save('./features_data/labels.npy', np.array(self.training_labels))
                np.save('./features_data/Distance_abstract_train.npy', np.array(Distance_abstract))
                np.save('./features_data/Distance_title_train.npy', np.array(Distance_title))
                np.save('./features_data/tfidf_distance_corpus_train.npy', np.array(tfidf_distance_corpus))
                np.save('./features_data/tfidf_distance_titles_train.npy', np.array(tfidf_distance_titles))
                np.save('./features_data/jaccard_und_train.npy', np.array(jaccard_und))
                np.save('./features_data/Resource_allocation_train.npy', np.array(Resource_allocation))
            else:
                np.save('./features_data/shortest_path_dijkstra_test.npy', np.array(shortest_path_dijkstra))
                np.save('./features_data/shortest_path_dijkstra_und_test.npy', np.array(shortest_path_dijkstra_und))
                np.save('./features_data/comm_neighbors_test.npy', np.array(comm_neighbors))
                np.save('./features_data/no_edge_test.npy', np.array(no_edge))
                np.save('./features_data/overlap_title_test.npy', np.array(overlap_title))
                np.save('./features_data/temp_diff_test.npy', np.array(temp_diff))
                np.save('./features_data/comm_auth_test.npy', np.array(comm_auth))
                np.save('./features_data/num_inc_edges_test.npy', np.array(num_inc_edges))
                np.save('./features_data/Distance_abstract_test.npy', np.array(Distance_abstract))
                np.save('./features_data/Distance_title_test.npy', np.array(Distance_title))
                np.save('./features_data/tfidf_distance_corpus_test.npy', np.array(tfidf_distance_corpus))
                np.save('./features_data/tfidf_distance_titles_test.npy', np.array(tfidf_distance_titles))
                np.save('./features_data/jaccard_und_test.npy', np.array(jaccard_und))
                np.save('./features_data/Resource_allocation_test.npy', np.array(Resource_allocation))


    def load_features_all(self):

        shortest_path_dijkstra = np.load('./features_data/shortest_path_dijkstra_train.npy')
        shortest_path_dijkstra_und = np.load('./features_data/shortest_path_dijkstra_und_train.npy')
        comm_neighbors = np.load('./features_data/comm_neighbors_train.npy')
        no_edge = np.load('./features_data/no_edge_train.npy')
        overlap_title = np.load('./features_data/overlap_title_train.npy')
        temp_diff = np.load('./features_data/temp_diff_train.npy')
        comm_auth = np.load('./features_data/comm_auth_train.npy')
        num_inc_edges = np.load('./features_data/num_inc_edges_train.npy')
        Distance_abstract = np.load('./features_data/Distance_abstract_train.npy')
        Distance_title = np.load('./features_data/Distance_title_train.npy')
        tfidf_distance_corpus = np.load('./features_data/tfidf_distance_corpus_train.npy').reshape(61551)
        tfidf_distance_titles = np.load('./features_data/tfidf_distance_titles_train.npy').reshape(61551)
        jaccard_und = np.load('./features_data/jaccard_und_train.npy')
        Resource_allocation = np.load('./features_data/Resource_allocation_train.npy')
        self.train_features = np.array([overlap_title, temp_diff, comm_auth,
                                        num_inc_edges, Distance_abstract, Distance_title,
                                        shortest_path_dijkstra, shortest_path_dijkstra_und,
                                        comm_neighbors, no_edge, tfidf_distance_corpus, tfidf_distance_titles,
                                        jaccard_und, Resource_allocation]).T

        shortest_path_dijkstra = np.load('./features_data/shortest_path_dijkstra_test.npy')
        shortest_path_dijkstra_und = np.load('./features_data/shortest_path_dijkstra_und_test.npy')
        comm_neighbors = np.load('./features_data/comm_neighbors_test.npy')
        no_edge = np.load('./features_data/no_edge_test.npy')
        overlap_title = np.load('./features_data/overlap_title_test.npy')
        temp_diff = np.load('./features_data/temp_diff_test.npy')
        comm_auth = np.load('./features_data/comm_auth_test.npy')
        num_inc_edges = np.load('./features_data/num_inc_edges_test.npy')
        Distance_abstract = np.load('./features_data/Distance_abstract_test.npy')
        Distance_title = np.load('./features_data/Distance_title_test.npy')
        tfidf_distance_corpus = np.load('./features_data/tfidf_distance_corpus_test.npy').reshape(32648)
        tfidf_distance_titles = np.load('./features_data/tfidf_distance_titles_test.npy').reshape(32648)
        jaccard_und = np.load('./features_data/jaccard_und_test.npy')
        Resource_allocation = np.load('./features_data/Resource_allocation_test.npy')

        self.test_features = np.array([overlap_title, temp_diff, comm_auth,
                                        num_inc_edges, Distance_abstract, Distance_title,
                                        shortest_path_dijkstra, shortest_path_dijkstra_und,
                                        comm_neighbors, no_edge, tfidf_distance_corpus, tfidf_distance_titles,
                                        jaccard_und, Resource_allocation[:]]).T

        self.training_labels = np.array([int(element[2]) for element in self.training_set_reduced])
        return(self.train_features, self.training_labels, self.test_features)


    def add_feature_transformer(self):

        for type_data in [self.training_set_reduced, self.testing_set]:
            Distance_abstract = []
            Distance_title =[]
            counter = 0
            for i in range(len(type_data)):
                # for i in range(100):
                source = type_data[i][0]
                target = type_data[i][1]

                source_info = [element for element in self.node_info if element[0] == source][0]
                target_info = [element for element in self.node_info if element[0] == target][0]

                # convert to lowercase and tokenize
                source_title = source_info[2].lower().split(" ")
                print(source_title)
                # remove stopwords
                source_title = [token for token in source_title if token not in self.stpwds]
                source_title = [self.stemmer.stem(token) for token in source_title]
                print(source_title)


                source_title_glove = self.decomposition.apply(source_info[2].lower().split(" "))
                target_title_glove = self.decomposition.apply(target_info[2].lower().split(" "))

                source_abstract_glove = self.decomposition.apply(source_info[5].lower().split(" "))
                target_abstract_glove = self.decomposition.apply(target_info[5].lower().split(" "))

                distance_title = scipy.spatial.distance.euclidean(source_title_glove, target_title_glove)
                distance_abstract = scipy.spatial.distance.euclidean(source_abstract_glove, target_abstract_glove)
                Distance_abstract.append(distance_abstract)
                Distance_title.append(distance_title)

                counter += 1
                if counter % 1000 == True:
                    print(counter, "testing examples processsed")




    def load_glove_model(self, glove_file):
        """
        :param glove_file: adress of glove file
        :return:
        """
        print("Loading Glove Model")
        f = open(glove_file, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        return model

    def get_glove_matrix(self, article):
        """
        Get the Glove of an article
        :param article
        """
        N = 0
        vector = np.zeros(100)
        for k, word in enumerate(article):
            try:
                N += 1
                vector = vector + self.glove[word]
            except Exception:
                vector = vector + self.glove['unk']
        return vector / N
