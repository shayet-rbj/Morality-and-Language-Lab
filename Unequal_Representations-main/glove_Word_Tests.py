"""
Script that runs all GloVe embedding tests for Unequal Representation: Analyzing Intersectional Biases in Word Embeddings 
Using Representational Similarity Analysis by Michael Lepori, presented at COLING 2020
"""

import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.descriptivestats import sign_test
import glove_utils.utils as utils
import random
import pandas as pd


def preprocess_data(corpus):
    # Preprocess datasets
    sent_list = []
    for sent in corpus:
        sent = sent.lower()
        sent = sent.replace('.', '')
        sent_list.append(sent)
    return sent_list


def get_glove_embeds(glove_path, dim, corpus):
    # Get glove embeddings of all terms in the corpus
    word2idx, idx2word = utils.create_word_idx_matrices([corpus])
    print("word idx matrices created")
    glove = utils.create_embedding_dictionary(glove_path, dim, word2idx, idx2word)
    print("glove matrices created")

    glove_embeds = []

    for word in corpus:
        glove_embeds.append(glove[word2idx[word]])
        
    return np.array(glove_embeds)


def make_concept(samps, grp1, grp2, attr):
    # Make the hypothesis models encoding the hypothesized representational geometry.
    grp1_attr = np.zeros((len(samps), len(samps)))
    grp2_attr = np.zeros((len(samps), len(samps)))

    for i in range(len(samps)):
        sent1 = samps[i]
        for j in range(len(samps)):
            sent2 = samps[j]
            # Represents the hypothesis that group 1 has the attribute/concept under study, and group 2 does not
            if ((sent1 in attr or sent1 in grp1) and (sent2 in attr or sent2 in grp1)) or ((sent1 in grp2) and (sent2 in grp2)):
                # Dissimilarity matrices, so 0 means perfect alignment
                grp1_attr[i][j] = 0
            else:
                grp1_attr[i][j] = 1
            # Represents the hypothesis that group 2 has the attribute/concept under study, and group 1 does not
            if ((sent1 in attr or sent1 in grp2) and (sent2 in attr or sent2 in grp2)) or ((sent1 in grp1) and (sent2 in grp1)):
                # Dissimilarity matrices, so 0 means perfect alignment
                grp2_attr[i][j] = 0
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
        glove_embeds= get_glove_embeds("glove_utils/glove/glove.6B.300d.txt", 300, preprocess_data(corpus))
        print("glove embeds generated")

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
            samp_glove = glove_embeds[sample]
            grp1_attr_model, grp2_attr_model = make_concept(samp_sentences, group1, group2, concept)

            # 1 - spearman's r similarity matrix to make dissimilarity matrix
            glove_sim = np.ones(samp_glove.shape[0]) - spearmanr(samp_glove, axis=1)[0]
            # Take upper triangle
            glove_sim = glove_sim[np.triu_indices(samp_glove.shape[0], 1)].reshape(-1)

            grp1_attr_model = grp1_attr_model[np.triu_indices(samp_glove.shape[0], 1)].reshape(-1)
            grp2_attr_model = grp2_attr_model[np.triu_indices(samp_glove.shape[0], 1)].reshape(-1)

            # Append representational similarity for group 1 and group 2
            rsa_grp1.append(spearmanr([glove_sim, grp1_attr_model], axis=1)[0])
            rsa_grp2.append(spearmanr([glove_sim, grp2_attr_model], axis=1)[0])

        print(f'RSA {grp1_name} {attr_name}: {np.mean(rsa_grp1)} STD: {np.std(rsa_grp1)}')
        print(f'RSA {grp2_name} {attr_name}: {np.mean(rsa_grp2)} STD: {np.std(rsa_grp2)}')

        # Significance test of differences between group 1 RSA and group 2 RSA
        print(f'Sign Test {grp1_name} vs. {grp2_name}: {sign_test(np.array(rsa_grp1) - np.array(rsa_grp2))[1]}\n')