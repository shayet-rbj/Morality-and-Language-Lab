"""
Script that runs all single-word BERT embedding tests for Unequal Representation: Analyzing Intersectional Biases in Word Embeddings 
Using Representational Similarity Analysis by Michael Lepori, presented at COLING 2020
"""

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("..")
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from statsmodels.stats.descriptivestats import sign_test
import random
import pandas as pd


def preprocess_data(corpus, concepts):
    # Preprocess dataset for BERT
    bert_list = []
    for sent in corpus:
        if sent not in concepts:
            sent = sent[0].upper() + sent[1:]
        else:
            sent = sent.lower()
        sent = sent.replace('.', '')
        bert_list.append("[CLS] " + sent + " [SEP]")
    return bert_list


def get_bert_embeds(bert_sents):
    # Get BERT-base-cased model and extract embeddings from it
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertModel.from_pretrained('bert-base-cased')
    model.eval()

    bert_embeds = []

    for sent in bert_sents:

        # BERT encoding stuff
        encoding = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

        segment_ids = [1] * len(encoding)
        tokens_tensor = torch.tensor([encoding])
        segments_tensor = torch.tensor([segment_ids])

        # Always want the encoding of the word, not the CLS or SEP token
        idx = 1

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensor)
            
        # Take embedding from final layer of BERT base cased (layer 12)
        sentence_encoding = encoded_layers[11].reshape(len(tokens_tensor[0]), -1)[idx].reshape(-1)
        bert_embeds.append(sentence_encoding.numpy())

    return np.stack(bert_embeds, axis=0)


def make_concept(samps, grp1, grp2, attr):
    # Make the hypothesis models encoding the hypothesized representational geometry.
    grp1_attr = np.zeros((len(samps), len(samps)))
    grp2_attr = np.zeros((len(samps), len(samps)))

    for i in range(len(samps)):
        sent1 = samps[i]
        for j in range(len(samps)):
            sent2 = samps[j]
            if ((sent1 in attr or sent1 in grp1) and (sent2 in attr or sent2 in grp1)) or ((sent1 in grp2) and (sent2 in grp2)):
                # Represents the hypothesis that group 1 has the attribute/concept under study, and group 2 does not
                grp1_attr[i][j] = 0
                # Dissimilarity matrices, so 0 means perfect alignment
            else:
                grp1_attr[i][j] = 1

            if ((sent1 in attr or sent1 in grp2) and (sent2 in attr or sent2 in grp2)) or ((sent1 in grp1) and (sent2 in grp1)):
                # Represents the hypothesis that group 2 has the attribute/concept under study, and group 1 does not
                grp2_attr[i][j] = 0
                # Dissimilarity matrices, so 0 means perfect alignment
            else:
                grp2_attr[i][j] = 1
    
    return grp1_attr, grp2_attr
    


if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(seed=9)
    random.seed(9)

    # Read data
    sheets = pd.ExcelFile('Data/Word_Data.xlsx')

    # Every datasheet defines 1 test
    for idx, name in enumerate(sheets.sheet_names):
        print(name)
        sheet = sheets.parse(name)
        group1 = list(sheet.iloc[:, 0].dropna())
        grp1_name = sheet.columns[0]
        group2 = list(sheet.iloc[:, 1].dropna())
        grp2_name = sheet.columns[1]
        concept = list(sheet.iloc[:, 2].dropna())
        attr_name = sheet.columns[2]

        # corpus is all words in dataset
        corpus = group1 + group2 + concept
        bert_embeds = get_bert_embeds(preprocess_data(corpus, concept))
        print("bert embeds generated")

        rsa_grp1 = []
        rsa_grp2 = []

        # Sample 100 different configurations
        samps = []
        while len(samps) < 100:

            # sample 10 elements of group 1, 10 elements of group 2, 10 attribute elements
            sample = list(np.random.choice(range(0, len(group1)), replace = False, size=10)) + list(np.random.choice(range(len(group1), len(group1) + len(group2)), replace = False, size=10)) + list(np.random.choice(range(len(group1) + len(group2), len(group1) + len(group2) + len(concept)), size=10))
        
            if list(sample) in samps:
                continue

            samps.append(sample)

            # Make hypothesis models, as well as reference models
            samp_sentences = np.array(corpus)[sample]
            samp_bert = bert_embeds[sample]
            grp1_attr_model, grp2_attr_model = make_concept(samp_sentences, group1, group2, concept)

            # 1 - spearman's r similarity matrix to make dissimilarity matrix
            bert_sim = np.ones(samp_bert.shape[0]) - spearmanr(samp_bert, axis=1)[0]
            # Take upper triangle
            bert_sim = bert_sim[np.triu_indices(samp_bert.shape[0], 1)].reshape(-1)
            grp1_attr_model = grp1_attr_model[np.triu_indices(samp_bert.shape[0], 1)].reshape(-1)
            grp2_attr_model = grp2_attr_model[np.triu_indices(samp_bert.shape[0], 1)].reshape(-1)

            # Append representational similarity for group 1 and group 2
            rsa_grp1.append(spearmanr([bert_sim, grp1_attr_model], axis=1)[0])
            rsa_grp2.append(spearmanr([bert_sim, grp2_attr_model], axis=1)[0])

        print(f'RSA {grp1_name} {attr_name}: {np.mean(rsa_grp1)} STD: {np.std(rsa_grp1)}')
        print(f'RSA {grp2_name} {attr_name}: {np.mean(rsa_grp2)} STD: {np.std(rsa_grp2)}')

        # Significance test of differences between group 1 RSA and group 2 RSA
        print(f'Sign Test {grp1_name} vs. {grp2_name}: {sign_test(np.array(rsa_grp1) - np.array(rsa_grp2))[1]}\n')