from collections import deque

from sklearn import preprocessing

import kashgari
from kashgari.embeddings import NumericFeaturesEmbedding, BareEmbedding, StackedEmbedding, DirectEmbedding
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import chain


class Counterize:

    def __init__(self):
        self.dict = dict()
        self.size = 0

    def get(self, x):
        if x in self.dict:
            return self.dict[x]

        self.size += 1
        self.dict[x] = self.size

        return self.dict[x]


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def chunks(iterable, chunk_size=3, overlap=0):
    # we'll use a deque to hold the values because it automatically
    # discards any extraneous elements if it grows too large
    if chunk_size < 1:
        raise AttributeError("chunk size too small")
    if overlap >= chunk_size:
        raise AttributeError("overlap too large")
    queue = deque(maxlen=chunk_size)
    it = iter(iterable)
    i = 0
    try:
        # start by filling the queue with the first group
        for i in range(chunk_size):
            queue.append(next(it))
        while True:
            yield tuple(queue)
            # after yielding a chunk, get enough elements for the next chunk
            for i in range(chunk_size - overlap):
                queue.append(next(it))
    except StopIteration:
        # if the iterator is exhausted, yield any remaining elements
        i += overlap
        if i > 0:
            yield tuple(queue)[-i:]


class Kashgari:

    def __init__(self):
        self.model = None
        self.chunk_size = 100
        self.set_features_numeric = dict()
        self.set_features_text = dict()

    def prepare_data_fit(self, tokens, tags, chunk_size, overlap=10):
        text_list = []
        first_of_p_list = []
        tag_list = []

        buffer_text = []
        buffer_first_of_p = []
        buffer_tag = []

        text_features = set("token")
        numeric_features = set("first_of_p")

        self.set_features_numeric = dict()

        for doc, doc_tags in zip(tokens, tags):
            for token, tag in zip(doc, doc_tags):
                features = agregado(token, simple_features=True)
                buffer_text.append(features['token'])
                buffer_first_of_p.append('2' if features['first_of_p'] else '1')
                buffer_tag.append(tag)

                if len(buffer_text) > chunk_size:
                    text_list.append(buffer_text)
                    first_of_p_list.append(buffer_first_of_p)
                    tag_list.append(buffer_tag)
                    # Zerar
                    buffer_text = []
                    buffer_first_of_p = []
                    buffer_tag = []

            print("Processed doc")

        if len(buffer_text) >= 0:
            text_list.append(buffer_text)
            first_of_p_list.append(buffer_first_of_p)
            tag_list.append(buffer_tag)

        results = (text_list, first_of_p_list)
        return results, tag_list

    def prepare_data_predict(self, tokens, chunk_size):
        text_list = []
        first_of_p_list = []

        buffer_text = []
        buffer_first_of_p = []

        for token in tokens:
            features = agregado(token, simple_features=True)
            buffer_text.append(features['token'])
            buffer_first_of_p.append('2' if features['first_of_p'] else '1')

            if len(buffer_text) >= chunk_size:
                text_list.append(buffer_text)
                first_of_p_list.append(buffer_first_of_p)
                # Zerar
                buffer_text = []
                buffer_first_of_p = []

        if len(buffer_text) > 0:
            text_list.append(buffer_text)
            first_of_p_list.append(buffer_first_of_p)

        results = (text_list, first_of_p_list)

        return results

    def train(self, tokens, tags):

        x, y = self.prepare_data_fit(tokens, tags, chunk_size=self.chunk_size)

        text_embedding = BareEmbedding(task=kashgari.LABELING, sequence_length=self.chunk_size)
        first_of_p_embedding = NumericFeaturesEmbedding(feature_count=2,
                                                        feature_name='first_of_p',
                                                        sequence_length=self.chunk_size)

        stack_embedding = StackedEmbedding([
            text_embedding,
            first_of_p_embedding
        ])

        stack_embedding.analyze_corpus(x, y)

        from kashgari.tasks.labeling import BiLSTM_Model, BiLSTM_CRF_Model
        self.model = BiLSTM_CRF_Model(embedding=stack_embedding)
        self.model.fit(x, y, batch_size=1, epochs=20)

    def predict(self, tokens):
        import itertools
        results = []
        for doc in tokens:
            x = self.prepare_data_predict(doc, chunk_size=self.chunk_size)

            predicted = self.model.predict(x)
            x_list = list(itertools.chain.from_iterable(x[0]))
            predicted_unified = list(itertools.chain.from_iterable(predicted))
            predicted_truncated = predicted_unified[:len(doc)]

            print(f"len doc{len(doc)} | x_list{len(x_list)} |len predicted_unified{len(predicted_unified)} |len predicted_truncated{len(predicted_truncated)} |")
            results.append(predicted_unified[:len(doc)])

        return results


if __name__ == '__main__':
    text = [[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]
    label = ['B-Category', 'I-Category', 'B-ProjectName', 'I-ProjectName', 'I-ProjectName']

    text_list = [text] * 100
    label_list = [label] * 100

    SEQUENCE_LEN = 80

    # You can use WordEmbedding or BERTEmbedding for your text embedding
    bare_embedding = DirectEmbedding(task=kashgari.RAW_LABELING, sequence_length=SEQUENCE_LEN, embedding_size=3)
    #bare_embedding = BareEmbedding(task=kashgari.LABELING, sequence_length=SEQUENCE_LEN)


    x = (text_list)
    y = label_list
    bare_embedding.analyze_corpus(x, y)

    # Now we can embed with this stacked embedding layer
    # We can build any labeling model with this embedding

    from kashgari.tasks.labeling import BiLSTM_Model, BiLSTM_CRF_Model
    model = BiLSTM_CRF_Model(embedding=bare_embedding)
    model.fit(x, y, batch_size=1, epochs=3)

    print(model.predict(x))
    #print(model.predict_entities(x))