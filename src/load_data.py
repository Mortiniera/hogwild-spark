import numpy as np
from scipy.sparse import csr_matrix
from os import path
import settings as s

def line_to_topic(r):
    topic, doc_id, _ = r[0].strip().split(' ')
    return int(doc_id), set([topic])


def line_to_features(r):
    r = r[0].strip().split(' ')
    features = [feature.split(':') for feature in r[2:]]
    col_idx = np.array([0] + [int(idx) + 1 for idx, _ in features])
    row_idx = np.array([0]*(len(features) + 1))
    data = np.array([1.] + [float(value) for _, value in features])
    return int(r[0]), csr_matrix((data, (row_idx, col_idx)), shape=(1, s.dim))


class DataLoader:
    def __init__(self, spark):
        self.spark = spark
        self.doc_category = self.read_category()
        self.training_set, self.validation_set = self.read_train_val_data()
        self.training_set.cache().count()
        self.validation_set.cache().count()
        self.test_set = self.read_test_data()

    def read_category(self, category='CCAT'):
        topics_lines = self.spark.read.text(
            path.join(s.path, 'datasets/rcv1-v2.topics.qrels')).rdd.map(line_to_topic)
        return topics_lines.reduceByKey(lambda x, y: x | y).map(
            lambda x: (x[0], 1) if category in x[1] else (x[0], -1)).cache()

    def read_train_val_data(self):
        train_lines = self.spark.read.text(path.join(
            s.path, 'datasets/lyrl2004_vectors_train.dat')).rdd.map(line_to_features)
        return train_lines.join(self.doc_category).map(lambda x: (
            x[1][0], x[1][1])).randomSplit([1 - s.validation_frac, s.validation_frac], seed=42)

    def read_test_data(self):
        test_lines = []
        for i in range(4):
            test_lines.append(self.spark.read.text(path.join(s.path, 'datasets/lyrl2004_vectors_test_pt'+str(i)+'.dat')))
        test_lines = test_lines[0].union(test_lines[1]).union(test_lines[2]).union(test_lines[3])
        return test_lines.rdd.map(line_to_features).join(self.doc_category).map(lambda x : (x[1][0], x[1][1])).cache()
