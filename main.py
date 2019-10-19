import os
import pickle
from pathlib import Path

import torch
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
import kashgari
from kashgari.embeddings import NumericFeaturesEmbedding, BareEmbedding, StackedEmbedding, DirectEmbedding
import random
import itertools
# init embedding
from kashgari.processors import ClassificationProcessor
from kashgari.processors.direct_classification_processor import DirectClassificationProcessor
from sklearn.metrics import classification_report
from kashgari.tasks.classification import CNN_GRU_Model, DPCNN_Model

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
                arg1 = partes[1].strip('"')
                rel = partes[2].strip('"')
                arg2 = partes[3].strip('"')
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
            # print(extraction)
            if any(len(x) < 1 for x in extraction.values()):
                extraction["invalid_format"] = True
                continue
            else:
                extraction["invalid_format"] = False

            extraction_to_embeddings(extraction, embeddings=item['bert_sentence'])
    return input_dict


def classificar(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
                folds_spanish):
    from kashgari.tasks.classification import CNNGRUModel, DPCNN_Model, BLSTMModel, CNNModel


    SEQUENCE_LEN = embedding.embedding_length
    # bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=SEQUENCE_LEN, embedding_size=3072)
    NUMBER_OF_EPOCHS = 50
    for model_type in [CNN_GRU_Model]:
        model_name = str(model_type).split(".")[-1].split("'")[0]

        # Vamos fazer o K-Fold agora
        # for k in range(len(folds_english)):
        #     print(f"Processing fold_{k}_{model_name}.model")
        #     x_all = []
        #     y_all = []
        #     # english
        #
        #     x_en, y_en = extractions_to_flat(dict_with_emmedings_en, list(itertools.chain.from_iterable(folds_english[k][0])))
        #     x_all.extend(x_en)
        #     y_all.extend(y_en)
        #     # Portuguese
        #     x_pt, y_pt = extractions_to_flat(dict_with_emmedings_pt, list(itertools.chain.from_iterable(folds_portuguese[k][0])))
        #     x_all.extend(x_pt)
        #     y_all.extend(y_pt)
        #     # Spanish
        #     x_es, y_es = extractions_to_flat(dict_with_emmedings_es, list(itertools.chain.from_iterable(folds_spanish[k][0])))
        #     x_all.extend(x_es)
        #     y_all.extend(y_es)
        #
        #     #bare_embedding = BareEmbedding(task=kashgari.CLASSIFICATION, sequence_length=SEQUENCE_LEN,
        #     #                               embedding_size=embedding_size,
        #     #                               processor=ClassificationProcessor(transform_input=False))
        #     bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=3,
        #                                                                  embedding_size=SEQUENCE_LEN,
        #                     processor=DirectClassificationProcessor(transform_input=False))
        #     bare_embedding.analyze_corpus(x_all, y_all)
        #
        #     model = model_type(embedding=bare_embedding)
        #     model.fit(x_all, y_all, batch_size=1, epochs=NUMBER_OF_EPOCHS)
        #     model.save(f"fold_{k}_{model_name}.model")


        # O primeiro eh o conjunto completo
        print("Processing Zero-shot en+es")
        x_all = []
        y_all = []
        x_en, y_en = extractions_to_flat(dict_with_emmedings_en)
        x_all.extend(x_en)
        y_all.extend(y_en)
        x_es, y_es = extractions_to_flat(dict_with_emmedings_es)
        x_all.extend(x_es)
        y_all.extend(y_es)

        bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=3,
                                         embedding_size=SEQUENCE_LEN,
                                         processor=DirectClassificationProcessor(transform_input=False))
        bare_embedding.analyze_corpus(x_all, y_all)

        model = model_type(embedding=bare_embedding)
        model.fit(x_en, y_en, batch_size=1, epochs=NUMBER_OF_EPOCHS)

        model.save(f"en_all_cnn_{model_name}_en_es.model")

        print("Processing Zero-shot en+pt")
        x_all = []
        y_all = []
        x_en, y_en = extractions_to_flat(dict_with_emmedings_en)
        x_all.extend(x_en)
        y_all.extend(y_en)
        x_pt, y_pt = extractions_to_flat(dict_with_emmedings_pt)
        x_all.extend(x_pt)
        y_all.extend(y_pt)

        bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=3,
                                         embedding_size=SEQUENCE_LEN,
                                         processor=DirectClassificationProcessor(transform_input=False))
        bare_embedding.analyze_corpus(x_all, y_all)

        model = model_type(embedding=bare_embedding)
        model.fit(x_en, y_en, batch_size=1, epochs=NUMBER_OF_EPOCHS)

        model.save(f"en_all_cnn_{model_name}_en_pt.model")


def extractions_to_flat(dict_with_emmedings, indexes=None):
    x = []
    y = []
    count = 0
    if indexes is None:
        indexes = dict_with_emmedings.keys()

    for pos in indexes:
        item = dict_with_emmedings[pos]
        count += 1
        #if count > 1000:
        #    break
        for extraction in item['extractions']:
            #total_representation = []
            if extraction["invalid_format"]:
                print(f"Extracton have the wrong format ({extraction})")
                continue

            #total_representation.extend(extraction['arg1_vec'])
            #total_representation.extend(extraction['rel_vec'])
            #total_representation.extend(extraction['arg2_vec'])
            x.append([extraction['arg1_vec'],extraction['rel_vec'], extraction['arg2_vec'] ])

            if extraction['valid'] == 'Arafat':  # Bug no dataset em ingles
                extraction['valid'] = 0
            y.append(int(extraction['valid']))
    return x, y

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
            # print(f"Achei {first_to_find} na pos {pos}")
            start_match = pos

            for pos_fim in range(pos, len(embeddings.tokens)):
                if (last_to_find == clean_word(embeddings.tokens[pos_fim].text)) and (
                        pos_fim - start_match >= len(string_to_find) - 2):
                    # print(f"Achei o FIM {last_to_find} na pos {pos_fim}")
                    end_match = pos_fim
                    break
            if (len(string_to_find) == 1 or end_match > 0) and (
                    (abs((end_match - start_match) - len(string_to_find)) < 10) or (
                    (end_match - start_match) > len(string_to_find))):
                break

    tokens = []

    if (len(string_to_find) > 1 and end_match == 0) or (
            (abs((end_match - start_match) - len(string_to_find)) > 10) and (
            (end_match - start_match) < len(string_to_find))):
        print(
            f"[ERROR] Nao encontrei {string_to_find} sentenca {embeddings} | first_to_find{first_to_find} last_to_find{last_to_find}")
        sentence = Sentence(' '.join(string_to_find))
        # embed words in sentence
        result = embedding.embed(sentence)[0]
        for token in result.tokens:
            tokens.append(token)

    else:
        for pos in range(start_match, end_match + 1):
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
    print(classification_report(y_true_en, y_predicted_en, digits=6))

    print("---- Portuguese ----")
    y_true_pt = []
    for pos, doc in docs_pt.items():
        for extraction in doc['extractions']:
            y_true_pt.append(extraction['valid'])
    y_predicted_pt = ['1'] * len(y_true_pt)
    print(classification_report(y_true_pt, y_predicted_pt, digits=6))

    print("---- Spanish ----")
    y_true_es = []
    for pos, doc in docs_es.items():
        for extraction in doc['extractions']:
            y_true_es.append(extraction['valid'])
    y_predicted_es = ['1'] * len(y_true_es)
    print(classification_report(y_true_es, y_predicted_es, digits=6))

def generate_classification_report(model_lang, model_name, predictions, true_y):

    with open(f"{model_lang}{model_name}.txt", "a") as file_out:

        for precision_at in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]:
            predictions_ajusted = [x['label'] if x['confidence'] > precision_at else '1' for x in predictions]

            print(f"{model_lang}@{precision_at} - {model_name}")
            report = classification_report([str(y) for y in true_y], predictions_ajusted, output_dict=True)
            print(classification_report([str(y) for y in true_y], predictions_ajusted))
            file_out.write(f"{precision_at},{report['accuracy']},{report['1']['precision']},{report['1']['recall']},{report['1']['f1-score']},{report['1']['support']},{report['0']['precision']},{report['0']['recall']},{report['0']['f1-score']},{report['0']['support']}\n")


def evaluate(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
                folds_spanish):
    #for model_type in [CNN_GRU_Model, DPCNN_Model]:
    for model_type in [CNN_GRU_Model]:
        model_name = str(model_type).split(".")[-1].split("'")[0]

        # Zero-shot
        model = kashgari.utils.load_model(f"en_all_cnn_{model_name}_en_es.model")

        x_pt_zero, y_pt_zero = extractions_to_flat(dict_with_emmedings_pt)
        y_pt_pred_top_k = model.predict_top_k_class(x_pt_zero, top_k=2)
        generate_classification_report('pt-zero-shot', "Zero-shot", y_pt_pred_top_k, y_pt_zero)

        model = kashgari.utils.load_model(f"en_all_cnn_{model_name}_en_pt.model")

        x_es_zero, y_es_zero = extractions_to_flat(dict_with_emmedings_es)
        y_es_pred_top_k = model.predict_top_k_class(x_es_zero, top_k=2)
        generate_classification_report('es-zero-shot', "Zero-shot", y_es_pred_top_k, y_es_zero)

        # Vamos fazer o K-Fold agora
        for k in range(len(folds_english)):
            file_name_model = f"fold_{k}_{model_name}.model"
            print(file_name_model)
            model = kashgari.utils.load_model(file_name_model)

            print(f"English - {file_name_model}")
            x_en, y_en = extractions_to_flat(dict_with_emmedings_en, folds_english[k][1])
            y_en_pred_top_k = model.predict_top_k_class(x_en, top_k=2)
            generate_classification_report('en', model_name, y_en_pred_top_k, y_en)

            print(f"Portuguese - {file_name_model}")
            x_pt, y_pt = extractions_to_flat(dict_with_emmedings_pt, folds_portuguese[k][1])
            y_pt_pred_top_k = model.predict_top_k_class(x_pt, top_k=2)
            generate_classification_report('pt', model_name, y_pt_pred_top_k, y_pt)

            print(f"Spanish - {file_name_model}")
            x_es, y_es = extractions_to_flat(dict_with_emmedings_es, folds_spanish[k][1])
            y_es_pred_top_k = model.predict_top_k_class(x_es, top_k=2)
            generate_classification_report('es', model_name, y_es_pred_top_k, y_es)


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

    if not os.path.exists('folds_english.pickle'):
        folds_english = kfoldcv([x for x in dict_with_emmedings_en.keys()], k=5)
        with open('folds_english.pickle', 'wb') as handle:
            pickle.dump(folds_english, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_english.pickle', 'rb') as handle:
            folds_english = pickle.load(handle)

    if not os.path.exists('folds_portuguese.pickle'):
        folds_portuguese = kfoldcv([x for x in dict_with_emmedings_pt.keys()], k=5)
        with open('folds_portuguese.pickle', 'wb') as handle:
            pickle.dump(folds_portuguese, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_portuguese.pickle', 'rb') as handle:
            folds_portuguese = pickle.load(handle)

    if not os.path.exists('folds_spanish.pickle'):
        folds_spanish = kfoldcv([x for x in dict_with_emmedings_es.keys()], k=5)
        with open('folds_spanish.pickle', 'wb') as handle:
            pickle.dump(folds_spanish, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_spanish.pickle', 'rb') as handle:
            folds_spanish = pickle.load(handle)

    #classificar(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
    #            folds_spanish)

    evaluate(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
                 folds_spanish)
