import os
import pickle
from pathlib import Path

import torch
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
import kashgari
from kashgari.embeddings import NumericFeaturesEmbedding, BareEmbedding, StackedEmbedding, DirectEmbedding
import random

# init embedding
embedding = BertEmbeddings("bert-base-multilingual-cased")


def carregar_gamalho():

    # English dataset
    dataset_en = dict()

    en = Path("Dataset/gamalho/en/sentences.txt")
    with open(en, 'r', encoding='utf-8') as f_en:
        for line in f_en:
            pos, phase = line.split('\t')
            dataset_en[int(pos)] = {"phase": phase,
                               "extractions": []
                               }

    en = Path("Dataset/gamalho/en/extractions-all-labeled.txt")
    with open(en, 'r', encoding='utf-8') as f_en:
        for line in f_en:
            if '\t' in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1= partes[1].strip('"')
                rel= partes[2].strip('"')
                arg2= partes[3].strip('"')
                valid = partes[-1]

                dataset_en[pos]['extractions'].append({"arg1": arg1,
                                                       "rel": rel,
                                                       "arg2": arg2,
                                                       "valid": valid.strip()})

    # Portuguese dataset
    dataset_pt = dict()

    pt = Path("Dataset/gamalho/pt/sentences.txt")
    with open(pt, 'r', encoding='utf-8') as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phase = line.split('\t', 1)
            dataset_pt[int(pos)] = {"phase": phase.strip(),
                               "extractions": []
                               }

    pt = Path("Dataset/gamalho/pt/argoe-pt-labeled.csv")
    with open(pt, 'r', encoding='utf-8') as f_pt:
        for line in f_pt:
            if '\t' in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = partes[1].strip('"')
                rel = partes[2].strip('"')
                arg2 = partes[3].strip('"')
                valid = partes[-1]

                dataset_pt[pos]['extractions'].append({"arg1": arg1,
                                                       "rel": rel,
                                                       "arg2": arg2,
                                                       "valid": valid.strip()})

        # Spanish dataset
        dataset_es = dict()

        es = Path("Dataset/gamalho/es/sentences.txt")
        with open(es, 'r', encoding='utf-8') as f_es:
            for line in f_es:
                line = line.strip()
                pos, phase = line.split('\t', 1)
                dataset_es[int(pos)] = {"phase": phase.strip(),
                                        "extractions": []
                                        }

        es = Path("Dataset/gamalho/es/extractions-all-labeled.txt")
        with open(es, 'r', encoding='utf-8') as f_es:
            for line in f_es:
                if '\t' in line:
                    partes = line.split("\t")
                    pos = int(partes[0])
                    arg1 = partes[1].strip('"')
                    rel = partes[2].strip('"')
                    arg2 = partes[3].strip('"')
                    valid = partes[-1]

                    dataset_es[pos]['extractions'].append({"arg1": arg1,
                                                           "rel": rel,
                                                           "arg2": arg2,
                                                           "valid": valid.strip()})

    return dataset_en, dataset_pt, dataset_es


def gerar_emmbedings(input_dict):

    for pos, item in input_dict.items():

        # create a sentence
        sentence = Sentence(item['phase'])

        # embed words in sentence
        result = embedding.embed(sentence)

        item['bert_sentence'] = result[0]


        for extraction in item['extractions']:
            #print(extraction)
            if any(len(x) < 1 for x in extraction.values()):
                extraction["invalid_format"] = True
                continue
            else:
                extraction["invalid_format"] = False


            extraction_to_embeddings(extraction, embeddings=item['bert_sentence'])
    return input_dict


def classificar(input_en, input_pt, input_es):
    from kashgari.tasks.classification import CNNGRUModel, DPCNN_Model, BLSTMModel, CNNModel

    # Vamos classificar agora?
    x_en = []
    y_en = []

    x_all = []
    y_all = []

    # english
    for pos, item in input_en.items():
        for extraction in item['extractions']:
            total_representation = []
            if extraction["invalid_format"]:
                print(f"Extracton have the wrong format ({extraction})")
                continue

            total_representation.extend(extraction['arg1_vec'])
            total_representation.extend(extraction['rel_vec'])
            total_representation.extend(extraction['arg2_vec'])
            x_en.append(total_representation)
            x_all.append(total_representation)

            if extraction['valid'] == 'Arafat': # Bug no dataset em ingles
                extraction['valid'] = 0
            y_en.append(int(extraction['valid']))
            y_all.append(int(extraction['valid']))
    del input_en
    # Portuguese
    for pos, item in input_pt.items():
        for extraction in item['extractions']:
            total_representation = []
            if extraction["invalid_format"]:
                print(f"Extracton have the wrong format ({extraction})")
                continue

            total_representation.extend(extraction['arg1_vec'])
            total_representation.extend(extraction['rel_vec'])
            total_representation.extend(extraction['arg2_vec'])
            x_all.append(total_representation)

            y_all.append(int(extraction['valid']))
    del input_pt
    # Spanish
    for pos, item in input_es.items():
        for extraction in item['extractions']:
            total_representation = []
            if extraction["invalid_format"]:
                print(f"Extracton have the wrong format ({extraction})")
                continue

            total_representation.extend(extraction['arg1_vec'])
            total_representation.extend(extraction['rel_vec'])
            total_representation.extend(extraction['arg2_vec'])
            x_all.append(total_representation)

            y_all.append(int(extraction['valid']))
    del input_es
    SEQUENCE_LEN = 3072 * 3
    #bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=SEQUENCE_LEN, embedding_size=3072)
    bare_embedding = BareEmbedding(task=kashgari.CLASSIFICATION, sequence_length=SEQUENCE_LEN, embedding_size=500)

    bare_embedding.analyze_corpus(x_en, y_en)

    model = CNNModel(embedding=bare_embedding)
    model.fit(x_en, y_en, batch_size=1, epochs=20)

    model.save("en_cnn.model")






def extraction_to_embeddings(extraction, embeddings):

    # Arg1
    partes_arg1 = find_sublist_match(embeddings, extraction['arg1'])
    partes_rel = find_sublist_match(embeddings, extraction['rel'])
    partes_arg2 = find_sublist_match(embeddings, extraction['arg2'])
    extraction['arg1_vec'] = tokens_to_document_vectors(partes_arg1)
    extraction['rel_vec'] = tokens_to_document_vectors(partes_rel)
    extraction['arg2_vec'] = tokens_to_document_vectors(partes_arg2)

    # Forma 2, vamos colocar o emmeding de cada pedaco
    # TODO


def tokens_to_document_vectors(tokens):
    word_embeddings = []
    for token in tokens:
        word_embeddings.append(token.get_embedding().unsqueeze(0))

    word_embeddings = torch.cat(word_embeddings, dim=0)
    embedding_length = embedding.embedding_length
    embedding_flex = torch.nn.Linear(
        embedding_length, embedding_length, bias=False
    )
    embedding_flex.weight.data.copy_(torch.eye(embedding_length))

    word_embeddings = embedding_flex(word_embeddings)

    pooled_embedding = torch.mean(word_embeddings, 0)

    return pooled_embedding.detach().numpy()

# word_embeddings = torch.cat(word_embeddings, dim=0)
    # torch.mean
    # embedding_flex = torch.nn.Linear(
    #     self.embedding_length, self.embedding_length, bias=False
    # )


def find_sublist_match(embeddings, string_to_find, start=0):
    start_match = 0
    end_match = 0
    string_to_find = [x.text for x in Sentence(string_to_find).tokens]
    if len(string_to_find) < 1:
        print("Vazio!")

    def clean_word(word):
        return word.strip("'").strip('"').strip('.')
    first_to_find = clean_word(string_to_find[0])
    last_to_find = clean_word(string_to_find[-1])
    for pos in range(start, len(embeddings.tokens)):
        if first_to_find == clean_word(embeddings.tokens[pos].text):
            #print(f"Achei {first_to_find} na pos {pos}")
            start_match = pos

            for pos_fim in range(pos, len(embeddings.tokens)):
                if (last_to_find == clean_word(embeddings.tokens[pos_fim].text)) and (pos_fim-start_match >= len(string_to_find)-2):
                    #print(f"Achei o FIM {last_to_find} na pos {pos_fim}")
                    end_match = pos_fim
                    break
            if (len(string_to_find) == 1 or end_match > 0) and ((abs((end_match-start_match)-len(string_to_find)) < 10) or ((end_match-start_match)>len(string_to_find))):
                break

    tokens = []

    if (len(string_to_find) > 1 and end_match == 0) or ((abs((end_match-start_match)-len(string_to_find)) > 10) and ((end_match-start_match)<len(string_to_find))):
        print(f"[ERROR] Nao encontrei {string_to_find} sentenca {embeddings} | first_to_find{first_to_find} last_to_find{last_to_find}")
        sentence = Sentence(' '.join(string_to_find))
        # embed words in sentence
        result = embedding.embed(sentence)[0]
        for token in result.tokens:
            tokens.append(token)

    else:
        for pos in range(start_match, end_match+1):
            tokens.append(embeddings.tokens[pos])

    return tokens

def kfoldcv(indices, k=10, seed=42):

    size = len(indices)
    subset_size = round(size / k)
    random.Random(seed).shuffle(indices)
    subsets = [indices[x:x + subset_size] for x in range(0, len(indices), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train, test))

    return kfolds

def report_performance(docs_en, docs_pt, docs_es):
    from sklearn.metrics import classification_report
    print("---- English ----")
    y_true_en = []
    for pos, doc in docs_en.items():
        for extraction in doc['extractions']:
            y_true_en.append(extraction['valid'])
    y_predicted_en = ['1'] * len(y_true_en)
    print(classification_report(y_true_en, y_predicted_en))

    print("---- Portuguese ----")
    y_true_pt = []
    for pos, doc in docs_pt.items():
        for extraction in doc['extractions']:
            y_true_pt.append(extraction['valid'])
    y_predicted_pt = ['1'] * len(y_true_pt)
    print(classification_report(y_true_pt, y_predicted_pt))

    print("---- Spanish ----")
    y_true_es = []
    for pos, doc in docs_es.items():
        for extraction in doc['extractions']:
            y_true_es.append(extraction['valid'])
    y_predicted_es = ['1'] * len(y_true_es)
    print(classification_report(y_true_es, y_predicted_es))


if __name__ == '__main__':
    print("1 - Reading Dataset")
    docs_en, docs_pt, docs_es = carregar_gamalho()

    print("1.1 - Dataset performance")
    report_performance(docs_en, docs_pt, docs_es)

    print("2 - Generating Emmbedings")
    print("2.1 - Processing english")
    if not os.path.exists('processed_en.pickle'):
        dict_with_emmedings_en = gerar_emmbedings(docs_en)
        with open('processed_en.pickle', 'wb') as handle:
            pickle.dump(dict_with_emmedings_en, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("2.1 - English SKIPPED - Delete processed_en.pickle if you want to process again")
        with open('processed_en.pickle', 'rb') as handle:
            dict_with_emmedings_en = pickle.load(handle)

    print("2.2 - Processing Portuguese")
    if not os.path.exists('processed_pt.pickle'):
        dict_with_emmedings_pt = gerar_emmbedings(docs_pt)
        with open('processed_pt.pickle', 'wb') as handle:
            pickle.dump(dict_with_emmedings_pt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("2.2 - Portuguese SKIPPED - Delete processed_pt.pickle if you want to process again")
        with open('processed_pt.pickle', 'rb') as handle:
            dict_with_emmedings_pt = pickle.load(handle)

    print("2.3 - Processing Spanish")
    if not os.path.exists('processed_es.pickle'):
        dict_with_emmedings_es = gerar_emmbedings(docs_es)
        with open('processed_es.pickle', 'wb') as handle:
            pickle.dump(dict_with_emmedings_es, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("2.3 - Spanish SKIPPED - Delete processed_es.pickle if you want to process again")
        with open('processed_es.pickle', 'rb') as handle:
            dict_with_emmedings_es = pickle.load(handle)

    print("3 - Training classifier")

    classificar(dict_with_emmedings_en, dict_with_emmedings_pt, dict_with_emmedings_es)




