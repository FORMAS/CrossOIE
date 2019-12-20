import os
import pickle
from pathlib import Path

import torch
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings, XLMEmbeddings
import kashgari
from kashgari.embeddings import NumericFeaturesEmbedding, BareEmbedding, StackedEmbedding, DirectEmbedding
import random
import itertools
# init embedding
from kashgari.processors import ClassificationProcessor
from kashgari.processors.direct_classification_processor import DirectClassificationProcessor
from sklearn.metrics import classification_report, matthews_corrcoef
from kashgari.tasks.classification import CNN_GRU_Model, DPCNN_Model, CNN_Model, BiGRU_Model
from sklearn.isotonic import IsotonicRegression


EMBEDDING = BertEmbeddings("bert-base-multilingual-cased")
EMBEDDING_XLI = XLMEmbeddings("xlm-mlm-17-1280")


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


def carregar_lrec():
    # Portuguese dataset
    dataset_pt = dict()

    pt = Path("Dataset/lrec/200-sentences-pt-PUD.txt")
    with open(pt, 'r', encoding='utf-8') as f_pt:
        actual_pos = None
        for line in f_pt:
            line = line.strip()
            partes = line.split('\t')
            if len(partes) == 2:
                actual_pos = int(partes[0])
                dataset_pt[actual_pos] = {"phase": partes[1].strip(),
                                        "extractions": []
                                        }
            else:
                partes = line.split("\t")
                arg1 = partes[0].strip('"')
                rel = partes[1].strip('"')
                arg2 = partes[2].strip('"')
                valid = partes[-2]

                dataset_pt[actual_pos]['extractions'].append({"arg1": arg1,
                                                       "rel": rel,
                                                       "arg2": arg2,
                                                       "valid": valid.strip()})

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
        sentence_xli = Sentence(item['phase'])

        # embed words in sentence
        result = EMBEDDING.embed(sentence)

        result_xli = EMBEDDING_XLI.embed(sentence_xli)

        item['bert_sentence'] = result[0]
        item['xli_sentence'] = result_xli[0]

        for extraction in item['extractions']:
            # print(extraction)
            if any(len(x) < 1 for x in extraction.values()):
                extraction["invalid_format"] = True
                continue
            else:
                extraction["invalid_format"] = False

            extraction_to_embeddings(extraction, embeddings=item['bert_sentence'], embeddings_xli=item['xli_sentence'])
    return input_dict


def classificar(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
                folds_spanish):
    from kashgari.tasks.classification import CNNGRUModel, DPCNN_Model, BLSTMModel, CNNModel

    for emb_type in ['xli', 'bert']:

        if emb_type == 'bert':
            SEQUENCE_LEN = EMBEDDING.embedding_length
        else:
            SEQUENCE_LEN = EMBEDDING_XLI.embedding_length

        # bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=SEQUENCE_LEN, embedding_size=3072)
        NUMBER_OF_EPOCHS = 80
        BATCH_SIZE = 4
        for model_type in [CNN_Model, CNN_GRU_Model]:
            model_name = str(model_type).split(".")[-1].split("'")[0]

            # Vamos fazer o K-Fold agora
            for k in range(len(folds_english)):
                print(f"Processing fold_{k}_{emb_type}_{model_name}.model")
                x_all = []
                y_all = []
                x_all_test = []
                y_all_test = []
                # english

                x_en, y_en = extractions_to_flat(dict_with_emmedings_en, emb_type=emb_type,
                                                 indexes=list(itertools.chain.from_iterable(folds_english[k][0])))
                x_all.extend(x_en)
                y_all.extend(y_en)
                # Test
                x_en_test, y_en_test = extractions_to_flat(dict_with_emmedings_en, emb_type=emb_type,
                                                           indexes=folds_english[k][1])
                x_all_test.extend(x_en_test)
                y_all_test.extend(y_en_test)

                # Portuguese
                x_pt, y_pt = extractions_to_flat(dict_with_emmedings_pt, emb_type=emb_type,
                                                 indexes=list(itertools.chain.from_iterable(folds_portuguese[k][0])))
                x_all.extend(x_pt)
                y_all.extend(y_pt)
                # Test
                x_pt_test, y_pt_test = extractions_to_flat(dict_with_emmedings_pt, emb_type=emb_type,
                                                           indexes=folds_portuguese[k][1])
                x_all_test.extend(x_pt_test)
                y_all_test.extend(y_pt_test)

                # Spanish
                x_es, y_es = extractions_to_flat(dict_with_emmedings_es, emb_type=emb_type,
                                                 indexes=list(itertools.chain.from_iterable(folds_spanish[k][0])))
                x_all.extend(x_es)
                y_all.extend(y_es)

                x_es_test, y_es_test = extractions_to_flat(dict_with_emmedings_es, emb_type=emb_type,
                                                           indexes=folds_spanish[k][1])
                x_all_test.extend(x_es_test)
                y_all_test.extend(y_es_test)

                bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=3,
                                                 embedding_size=SEQUENCE_LEN,
                                                 processor=DirectClassificationProcessor(transform_input=False))
                bare_embedding.analyze_corpus(x_all, y_all)

                model = model_type(embedding=bare_embedding)
                model.fit(x_all, y_all, x_validate=x_all_test, y_validate=y_all_test, batch_size=BATCH_SIZE,
                          epochs=NUMBER_OF_EPOCHS)
                model.save(f"fold_{k}_{emb_type}_{model_name}.model")

            # O primeiro eh o conjunto completo
            print(f"Processing Zero-shot_{emb_type} en+es")
            x_all = []
            y_all = []
            x_en, y_en = extractions_to_flat(dict_with_emmedings_en, emb_type=emb_type)
            x_all.extend(x_en)
            y_all.extend(y_en)
            x_es, y_es = extractions_to_flat(dict_with_emmedings_es, emb_type=emb_type)
            x_all.extend(x_es)
            y_all.extend(y_es)

            bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=3,
                                             embedding_size=SEQUENCE_LEN,
                                             processor=DirectClassificationProcessor(transform_input=False))
            bare_embedding.analyze_corpus(x_all, y_all)

            model = model_type(embedding=bare_embedding)
            model.fit(x_en, y_en, batch_size=1, epochs=NUMBER_OF_EPOCHS)

            model.save(f"en_all_cnn_{model_name}_{emb_type}_en_es.model")

            print(f"Processing Zero-shot_{emb_type} en+pt")
            x_all = []
            y_all = []
            x_en, y_en = extractions_to_flat(dict_with_emmedings_en, emb_type=emb_type)
            x_all.extend(x_en)
            y_all.extend(y_en)
            x_pt, y_pt = extractions_to_flat(dict_with_emmedings_pt, emb_type=emb_type)
            x_all.extend(x_pt)
            y_all.extend(y_pt)

            bare_embedding = DirectEmbedding(task=kashgari.CLASSIFICATION, sequence_length=3,
                                             embedding_size=SEQUENCE_LEN,
                                             processor=DirectClassificationProcessor(transform_input=False))
            bare_embedding.analyze_corpus(x_all, y_all)

            model = model_type(embedding=bare_embedding)
            model.fit(x_en, y_en, batch_size=1, epochs=NUMBER_OF_EPOCHS)

            model.save(f"en_all_cnn_{model_name}_{emb_type}_en_pt.model")


def extractions_to_flat(dict_with_emmedings, emb_type, indexes=None):
    x = []
    y = []
    count = 0
    if indexes is None:
        indexes = dict_with_emmedings.keys()

    for pos in indexes:
        item = dict_with_emmedings[pos]
        count += 1
        # if count > 1000:
        #    break
        for extraction in item['extractions']:
            # total_representation = []
            if extraction["invalid_format"]:
                print(f"Extracton have the wrong format ({extraction})")
                continue

            if emb_type == 'bert':
                x.append([extraction['arg1_vec'], extraction['rel_vec'], extraction['arg2_vec']])
            else:
                x.append([extraction['arg1_vec_xli'], extraction['rel_vec_xli'], extraction['arg2_vec_xli']])

            if extraction['valid'] == 'Arafat':  # Bug no dataset em ingles
                extraction['valid'] = 0
            y.append(int(extraction['valid']))
    return x, y


def extraction_to_embeddings(extraction, embeddings, embeddings_xli):
    # Arg1
    partes_arg1 = find_sublist_match(embeddings, extraction['arg1'])
    partes_rel = find_sublist_match(embeddings, extraction['rel'])
    partes_arg2 = find_sublist_match(embeddings, extraction['arg2'])
    extraction['arg1_vec'] = tokens_to_document_vectors(partes_arg1)
    extraction['rel_vec'] = tokens_to_document_vectors(partes_rel)
    extraction['arg2_vec'] = tokens_to_document_vectors(partes_arg2)

    # Xli
    partes_arg1_xli = find_sublist_match(embeddings_xli, extraction['arg1'])
    partes_rel_xli = find_sublist_match(embeddings_xli, extraction['rel'])
    partes_arg2_xli = find_sublist_match(embeddings_xli, extraction['arg2'])
    extraction['arg1_vec_xli'] = tokens_to_document_vectors(partes_arg1_xli)
    extraction['rel_vec_xli'] = tokens_to_document_vectors(partes_rel_xli)
    extraction['arg2_vec_xli'] = tokens_to_document_vectors(partes_arg2_xli)


def tokens_to_document_vectors(tokens):
    word_embeddings = []
    for token in tokens:
        word_embeddings.append(token.get_embedding().unsqueeze(0))

    embedding_length = word_embeddings[0].shape[1]  # use the emmbeding first token size
    word_embeddings = torch.cat(word_embeddings, dim=0)
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
        if list(embeddings[0]._embeddings.keys())[0].startswith('bert'):
            result = EMBEDDING.embed(sentence)[0]
        else:
            result = EMBEDDING_XLI.embed(sentence)[0]

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
    from sklearn.metrics import classification_report, matthews_corrcoef
    print("---- English ----")
    y_true_en = []
    for pos, doc in docs_en.items():
        for extraction in doc['extractions']:

            if any(len(x) < 1 for x in extraction.values()):
                continue

            y_true_en.append(extraction['valid'])
    y_predicted_en = ['1'] * len(y_true_en)
    print(classification_report(y_true_en, y_predicted_en, digits=6))
    print("Matthews EN:")
    print(matthews_corrcoef(y_true_en, y_predicted_en))


    print("---- Portuguese ----")
    y_true_pt = []
    for pos, doc in docs_pt.items():
        for extraction in doc['extractions']:

            if any(len(x) < 1 for x in extraction.values()):
                continue

            y_true_pt.append(extraction['valid'])
    y_predicted_pt = ['1'] * len(y_true_pt)
    print(classification_report(y_true_pt, y_predicted_pt, digits=6))
    print("Matthews PT:")
    print(matthews_corrcoef(y_true_pt, y_predicted_pt))

    print("---- Spanish ----")
    y_true_es = []
    for pos, doc in docs_es.items():
        for extraction in doc['extractions']:

            if any(len(x) < 1 for x in extraction.values()):
                continue

            y_true_es.append(extraction['valid'])
    y_predicted_es = ['1'] * len(y_true_es)
    print(classification_report(y_true_es, y_predicted_es, digits=6))
    print("Matthews ES:")
    print(matthews_corrcoef(y_true_es, y_predicted_es))


def generate_classification_report(model_lang, model_name, predictions, true_y):
    ir = IsotonicRegression()
    #ir.fit(predictions, true_y)
    #results_en = ir.predict(predictions)

    with open(f"{model_lang}_{model_name}.txt", "a") as file_out:
        file_out.write(
            f"precision_at, accuracy, 1_precision, 1_recall, 1_f1-score, 1_support, 0_precision, 0_recall, 0_f1-score, 0_support\n")
        for precision_at in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]:
            predictions_ajusted = [x['label'] if x['confidence'] >= precision_at else '1' for x in predictions]

            print(f"{model_lang}@{precision_at} - {model_name}")
            report = classification_report([str(y) for y in true_y], predictions_ajusted, output_dict=True)
            #print(classification_report([str(y) for y in true_y], predictions_ajusted))
            print("Matthews:")
            print(matthews_corrcoef([str(y) for y in true_y], predictions_ajusted))

            file_out.write(
                f"{precision_at},{report['accuracy']},{report['1']['precision']},{report['1']['recall']},{report['1']['f1-score']},{report['1']['support']},{report['0']['precision']},{report['0']['recall']},{report['0']['f1-score']},{report['0']['support']}\n")


def evaluate(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
             folds_spanish):
    # for model_type in [CNN_GRU_Model, DPCNN_Model]:
    for emb_type in ['xli', 'bert']:
        for model_type in [CNN_Model, CNN_GRU_Model]:
            model_name = str(model_type).split(".")[-1].split("'")[0]

            # Vamos fazer o K-Fold agora
            y_en_all = []
            y_en_pred_all = []

            y_es_all = []
            y_es_pred_all = []

            y_pt_all = []
            y_pt_pred_all = []

            for k in range(len(folds_english)):
                file_name_model = f"fold_{k}_{emb_type}_{model_name}.model"
                print(file_name_model)
                model = kashgari.utils.load_model(file_name_model)

                print(f"English - _{emb_type} {file_name_model}")
                x_en, y_en = extractions_to_flat(dict_with_emmedings_en, emb_type=emb_type, indexes=folds_english[k][1])
                y_en_all.extend(y_en)
                y_en_pred_top_k = model.predict_top_k_class(x_en, top_k=2)
                y_en_pred_all.extend(y_en_pred_top_k)

                print(f"Portuguese - _{emb_type} {file_name_model}")
                x_pt, y_pt = extractions_to_flat(dict_with_emmedings_pt, emb_type=emb_type,
                                                 indexes=folds_portuguese[k][1])
                y_pt_all.extend(y_pt)
                y_pt_pred_top_k = model.predict_top_k_class(x_pt, top_k=2)
                y_pt_pred_all.extend(y_pt_pred_top_k)

                print(f"Spanish - _{emb_type} {file_name_model}")
                x_es, y_es = extractions_to_flat(dict_with_emmedings_es, emb_type=emb_type, indexes=folds_spanish[k][1])
                y_es_all.extend(y_es)
                y_es_pred_top_k = model.predict_top_k_class(x_es, top_k=2)
                y_es_pred_all.extend(y_es_pred_top_k)



            generate_classification_report(f'en_{emb_type}', model_name, y_en_pred_all, y_en_all)
            generate_classification_report(f'pt_{emb_type}', model_name, y_pt_pred_all, y_pt_all)
            generate_classification_report(f'es_{emb_type}', model_name, y_es_pred_all, y_es_all)

            # Zero-shot
            model = kashgari.utils.load_model(f"en_all_cnn_{model_name}_{emb_type}_en_es.model")

            x_pt_zero, y_pt_zero = extractions_to_flat(dict_with_emmedings_pt, emb_type=emb_type)
            y_pt_pred_top_k = model.predict_top_k_class(x_pt_zero, top_k=2)
            generate_classification_report(f'pt-zero-shot_{emb_type}', model_name, y_pt_pred_top_k, y_pt_zero)

            model = kashgari.utils.load_model(f"en_all_cnn_{model_name}_{emb_type}_en_pt.model")

            x_es_zero, y_es_zero = extractions_to_flat(dict_with_emmedings_es, emb_type=emb_type)
            y_es_pred_top_k = model.predict_top_k_class(x_es_zero, top_k=2)
            generate_classification_report(f'es-zero-shot_{emb_type}', model_name, y_es_pred_top_k, y_es_zero)


if __name__ == '__main__':
    print("1 - Reading Dataset")
    #docs_en, docs_pt, docs_es = carregar_gamalho()
    docs_en, docs_pt, docs_es = carregar_lrec()

    print("1.1 - Dataset performance")
    report_performance(docs_en, docs_pt, docs_es)

    print("2 - Generating Emmbedings")
    print("2.1 - Processing english")
    if not os.path.exists('processed_en_xli.pickle'):
        dict_with_emmedings_en = gerar_emmbedings(docs_en)
        with open('processed_en_xli.pickle', 'wb') as handle:
            pickle.dump(dict_with_emmedings_en, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("2.1 - English SKIPPED - Delete processed_en.pickle if you want to process again")
        with open('processed_en_xli.pickle', 'rb') as handle:
            dict_with_emmedings_en = pickle.load(handle)

    print("2.2 - Processing Portuguese")
    if not os.path.exists('processed_pt_xli.pickle'):
        dict_with_emmedings_pt = gerar_emmbedings(docs_pt)
        with open('processed_pt_xli.pickle', 'wb') as handle:
            pickle.dump(dict_with_emmedings_pt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("2.2 - Portuguese SKIPPED - Delete processed_pt.pickle if you want to process again")
        with open('processed_pt_xli.pickle', 'rb') as handle:
            dict_with_emmedings_pt = pickle.load(handle)

    print("2.3 - Processing Spanish")
    if not os.path.exists('processed_es_xli.pickle'):
        dict_with_emmedings_es = gerar_emmbedings(docs_es)
        with open('processed_es_xli.pickle', 'wb') as handle:
            pickle.dump(dict_with_emmedings_es, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("2.3 - Spanish SKIPPED - Delete processed_es.pickle if you want to process again")
        with open('processed_es_xli.pickle', 'rb') as handle:
            dict_with_emmedings_es = pickle.load(handle)

    print("3 - Training classifier")

    if not os.path.exists('folds_english_xli.pickle'):
        folds_english = kfoldcv([x for x in dict_with_emmedings_en.keys()], k=5)
        with open('folds_english_xli.pickle', 'wb') as handle:
            pickle.dump(folds_english, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_english_xli.pickle', 'rb') as handle:
            folds_english = pickle.load(handle)

    if not os.path.exists('folds_portuguese_xli.pickle'):
        folds_portuguese = kfoldcv([x for x in dict_with_emmedings_pt.keys()], k=5)
        with open('folds_portuguese_xli.pickle', 'wb') as handle:
            pickle.dump(folds_portuguese, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_portuguese_xli.pickle', 'rb') as handle:
            folds_portuguese = pickle.load(handle)

    if not os.path.exists('folds_spanish_xli.pickle'):
        folds_spanish = kfoldcv([x for x in dict_with_emmedings_es.keys()], k=5)
        with open('folds_spanish_xli.pickle', 'wb') as handle:
            pickle.dump(folds_spanish, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_spanish_xli.pickle', 'rb') as handle:
            folds_spanish = pickle.load(handle)

    classificar(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
                folds_spanish)

    #evaluate(dict_with_emmedings_en, folds_english, dict_with_emmedings_pt, folds_portuguese, dict_with_emmedings_es,
    #            folds_spanish)
