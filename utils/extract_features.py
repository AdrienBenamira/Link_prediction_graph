import random
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import scipy
import igraph

from utils.TransformerDecomposition import TransformerDecomposition

"""
nltk.download('punkt') # for tokenization
nltk.download('stopwords')"""



class features_dataset:
    def __init__(self, prepocess_all, ratio=0.10):
        """

        :param ratio:
        """
        if prepocess_all:
            self.glove = self.load_glove_model("./data_baseline/glove6B/glove.6B.100d.txt")
            self.decomposition = TransformerDecomposition()

        with open("./data_baseline/testing_set.txt", "r") as f:
            reader = csv.reader(f)
            testing_set = list(reader)
        self.testing_set = [element[0].split(" ") for element in testing_set]
        with open("./data_baseline/training_set.txt", "r") as f:
            reader = csv.reader(f)
            training_set = list(reader)
        self.training_set = [element[0].split(" ") for element in training_set]
        to_keep = random.sample(range(len(self.training_set)), k=int(round(len(self.training_set) * ratio)))
        self.training_set_reduced = [self.training_set[i] for i in to_keep]
        ###############################
        # beating the random baseline #
        ###############################
        # the following script gets an F1 score of approximately 0.66
        # data loading and preprocessing
        # the columns of the data frame below are:
        # (1) paper unique ID (integer)
        # (2) publication year (integer)
        # (3) paper title (string)
        # (4) authors (strings separated by ,)
        # (5) name of journal (optional) (string)
        # (6) abstract (string) - lowercased, free of punctuation except intra-word dashes
        with open("./data_baseline/node_information.csv", "r") as f:
            reader = csv.reader(f)
            self.node_info = list(reader)

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

        self.G = igraph.Graph(directed=True)
        self.G_und = None
        self.edges = [(element[0], element[1]) for element in self.training_set if element[2] == "1"]
        self.nodes = self.IDs
        self.G.add_vertices(self.nodes)
        self.G.add_edges(self.edges)
        self.G_und = self.G.as_undirected()


        self.stpwds = set(nltk.corpus.stopwords.words("english"))
        self.stemmer = nltk.stem.PorterStemmer()

    def prepocess_train_data(self):

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

        counter = 0
        for i in range(len(self.training_set_reduced)):
        #for i in range(100):
            source = self.training_set_reduced[i][0]
            target = self.training_set_reduced[i][1]

            source_info = [element for element in self.node_info if element[0] == source][0]
            target_info = [element for element in self.node_info if element[0] == target][0]


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


            #print(self.decomposition.apply(source_info[2].lower().split(" ")))

            counter += 1
            if counter % 1000 == True:
                print(counter, "training examples processsed")

        self.training_labels = np.array([int(element[2]) for element in self.training_set_reduced])

        np.save('./features_data/overlap_title_train.npy', np.array(overlap_title))
        np.save('./features_data/temp_diff_train.npy', np.array(temp_diff))
        np.save('./features_data/comm_auth_train.npy', np.array(comm_auth))
        np.save('./features_data/num_inc_edges_train.npy', np.array(num_inc_edges))
        np.save('./features_data/labels.npy', np.array(self.training_labels))
        np.save('./features_data/Distance_abstract_train.npy', np.array(Distance_abstract))
        np.save('./features_data/Distance_title_train.npy', np.array(Distance_title))

    def prepocess_test_data(self):

        # we need to compute the features_data for the testing set

        overlap_title = []
        temp_diff = []
        comm_auth = []
        num_inc_edges = []
        Distance_abstract = []
        Distance_title = []

        counter = 0
        for i in range(len(self.testing_set)):
        #for i in range(100):
            source = self.testing_set[i][0]
            target = self.testing_set[i][1]

            source_info = [element for element in self.node_info if element[0] == source][0]
            target_info = [element for element in self.node_info if element[0] == target][0]



            # convert to lowercase and tokenize
            source_title = source_info[2].lower().split(" ")
            # remove stopwords
            source_title = [token for token in source_title if token not in self.stpwds]
            source_title = [self.stemmer.stem(token) for token in source_title]

            target_title = target_info[2].lower().split(" ")
            target_title = [token for token in target_title if token not in self.stpwds]
            target_title = [self.stemmer.stem(token) for token in target_title]


            source_auth = source_info[3].split(",")
            target_auth = target_info[3].split(",")

            overlap_title.append(len(set(source_title).intersection(set(target_title))))
            temp_diff.append(int(source_info[1]) - int(target_info[1]))
            comm_auth.append(len(set(source_auth).intersection(set(target_auth))))


            num_inc_edges.append(len([element[1] for element in self.training_set if element[1] == target])
                                 + len([element[1] for element in self.testing_set if element[1] == target]))

            source_title_glove = self.get_glove_matrix(source_info[2].lower().split(" "))
            target_title_glove = self.get_glove_matrix(target_info[2].lower().split(" "))

            source_abstract_glove = self.get_glove_matrix(source_info[5].lower().split(" "))
            target_abstract_glove = self.get_glove_matrix(target_info[5].lower().split(" "))



            distance_title = scipy.spatial.distance.euclidean(source_title_glove, target_title_glove)
            distance_abstract = scipy.spatial.distance.euclidean(source_abstract_glove, target_abstract_glove)
            Distance_abstract.append(distance_abstract)
            Distance_title.append(distance_title)

            counter += 1
            if counter % 1000 == True:
                print(counter, "testing examples processsed")


        np.save('./features_data/overlap_title_test.npy', np.array(overlap_title))
        np.save('./features_data/temp_diff_test.npy', np.array(temp_diff))
        np.save('./features_data/comm_test.npy', np.array(comm_auth))
        np.save('./features_data/num_inc_edges_test.npy', np.array(num_inc_edges))
        np.save('./features_data/Distance_abstract_test.npy', np.array(Distance_abstract))
        np.save('./features_data/Distance_title_test.npy', np.array(Distance_title))


    def load_features_all(self):

        overlap_title_test = np.load('./features_data/overlap_title_test.npy')
        temp_diff_test = np.load('./features_data/temp_diff_test.npy')
        comm_test = np.load('./features_data/comm_test.npy')
        num_inc_edges_test = np.load('./features_data/num_inc_edges_test.npy')
        overlap_title_train = np.load('./features_data/overlap_title_train.npy')
        temp_diff_train = np.load('./features_data/temp_diff_train.npy')
        comm_train = np.load('./features_data/comm_auth_train.npy')
        num_inc_edges_train = np.load('./features_data/num_inc_edges_train.npy')
        Distance_abstract_train = np.load('./features_data/Distance_abstract_train.npy')
        Distance_title_train = np.load('./features_data/Distance_title_train.npy')
        Distance_abstract_test = np.load('./features_data/Distance_abstract_test.npy')
        Distance_title_test = np.load('./features_data/Distance_title_test.npy')
        inverse_shortest_distances_train = np.load('./features_data/inverse_shortest_distances_train.npy')
        inverse_shortest_distances_und_train = np.load('./features_data/inverse_shortest_distances_und_train.npy')
        comm_neighbors_train = np.load('./features_data/comm_neighbors_train.npy')
        no_edge_train = np.load('./features_data/no_edge_train.npy')
        inverse_shortest_distances_test = np.load('./features_data/inverse_shortest_distances_test.npy')
        inverse_shortest_distances_und_test = np.load('./features_data/inverse_shortest_distances_und_test.npy')
        comm_neighbors_test = np.load('./features_data/comm_neighbors_test.npy')
        no_edge_test = np.load('./features_data/no_edge_test.npy')




        self.train_features = np.array([overlap_title_train, temp_diff_train, comm_train,
                             num_inc_edges_train, Distance_abstract_train, Distance_title_train,
                            inverse_shortest_distances_train, inverse_shortest_distances_und_train,
                            comm_neighbors_train, no_edge_train]).T
        self.test_features = np.array([overlap_title_test, temp_diff_test, comm_test,
                                   num_inc_edges_test, Distance_abstract_test, Distance_title_test,
                                       inverse_shortest_distances_test, inverse_shortest_distances_und_test,
                                       comm_neighbors_test, no_edge_test]).T

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




    def add_feature_graph(self):

        for type_data in [self.testing_set, self.training_set_reduced]:

            inverse_shortest_distances = []
            inverse_shortest_distances_und = []
            comm_neighbors = []
            no_edge = []
            counter = 0
            for i in range(len(type_data)):
                # for i in range(100):
                source = type_data[i][0]
                target = type_data[i][1]

                index_source = self.IDs.index(source)
                index_target = self.IDs.index(target)

                self.G.delete_edges((index_source, index_target))

                inverse_shortest_distances.append(1. / (self.G.shortest_paths_dijkstra(source=source, target=target))[0][0]+0.)
                inverse_shortest_distances_und.append(1. / (self.G_und.shortest_paths_dijkstra(source=source, target=target))[0][0]+0.)
                self.G_und.add_edge(index_source, index_target)

                list_source = self.G.neighbors(source)
                list_target = self.G.neighbors(target)
                comm_neighbors.append(len(list(set(list_source).intersection(list_target))))

                no_edge.append(self.G.edge_disjoint_paths(index_source, index_target))

                counter += 1
                if counter % 1000 == True:
                    print(counter, "testing examples processsed")
            if type_data == self.training_set_reduced:
                np.save('./features_data/inverse_shortest_distances_train.npy', np.array(inverse_shortest_distances))
                np.save('./features_data/inverse_shortest_distances_und_train.npy', np.array(inverse_shortest_distances_und))
                np.save('./features_data/comm_neighbors_train.npy', np.array(comm_neighbors))
                np.save('./features_data/no_edge_train.npy', np.array(no_edge))
            else:
                np.save('./features_data/inverse_shortest_distances_test.npy', np.array(inverse_shortest_distances))
                np.save('./features_data/inverse_shortest_distances_und_test.npy',
                        np.array(inverse_shortest_distances_und))
                np.save('./features_data/comm_neighbors_test.npy', np.array(comm_neighbors))
                np.save('./features_data/no_edge_test.npy', np.array(no_edge))






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
