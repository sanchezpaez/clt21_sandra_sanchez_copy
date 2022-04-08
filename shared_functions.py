# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 06.04.2022

import operator

from nltk import word_tokenize


def read_lines_from_file(file_name):
    """
    Read lines from file, split by new line.
    :param file_name: file
    :return: list of str in the form
    ['the house', 'the blue house', 'the flower']
    """

    with open(file_name, encoding='utf-8') as file:
        text = file.read()
    text = text.strip().split('\n')
    return text


def clean_corpus_leave_punctuation(corpus, append_null=False):
    """
    Call tokenize() on all the sentences in the corpus.
    :param corpus: list of str
    :param append_null: Add token 'NULL' only to Quellsprache sentences.
    :return: list of lists of str
    """
    return [tokenize_not_remove_punctuation(
        sentence, append_null) for sentence in corpus]


def tokenize_not_remove_punctuation(sentence: str, append_null=False):
    """
    :param sentence: str. Example: 'the blue house'
    :param append_null: Add token 'NULL' only to Quellsprache sentences.
    :return: list of str. Example: ['NULL', 'the', 'blue', 'house']
    """
    tokens = word_tokenize(sentence)
    clean_tokens = [token.lower().strip() for token in tokens]
    if append_null:
        clean_tokens.insert(0, "NULL")
    return clean_tokens


def get_sentence_pairs(e_tokens, f_tokens):
    """
    Get list of tuples from two different lists.
    To every element in list A corresponds one element in list B.
    :param e_tokens: list of str
    :param f_tokens: list of str
    :return: list of tuples of list of str.
    Example: (['NULL', 'the', 'house'], ['la', 'maison'])
    """
    pairs = list(zip(e_tokens, f_tokens))
    return pairs


def calculate_word_alignments(probabilities_filename, parallel_corpus):
    """
    Use the trained model to calculate word alignments.
    Read probabilities file and convert each one in a dict entry.
    Get the word combination with the highest probability and get word indices.
    :param probabilities_filename: str with source_word\ttarget_word\tprobability
        Example: 'dreadful	comprobar	0.03160869924509357'
    :param parallel_corpus: list of tuples with
        ([source_sentence], [target_sentence])
    :return: list of str
        Every str represents the word alignments of one sentence.
        Example: in '0-0 1-2' a target sentence would have 2 positions,
        '0' has a source position 'NULL' and '1' is aligned to source
        word in index 2. Source word with index 1 would not have any
        translation in the target sentence.
    """
    probs_lines = read_lines_from_file(probabilities_filename)
    translation_probabilities = {}
    for prob_string in probs_lines:
        prob = prob_string.split('\t')
        translation_probabilities[(prob[0], prob[1])] = float(prob[2])
    sentences_alignments = []
    for src_sent, tgt_sent in parallel_corpus:
        sentence_alignments = []
        for t_w in tgt_sent:
            # Get translation probabilities for every sentence pair
            t_w_possible_translations = []
            for s_w in src_sent:
                t_w_index = tgt_sent.index(t_w) + 1
                # The index of every t_w position starts at 1,
                # since 0 is reserved for NULL in the source_sentence.
                s_w_index = src_sent.index(s_w)
                # Fetch the probabilities that are relevant for our word
                source_with_target = s_w, t_w
                try:
                    probability = translation_probabilities[source_with_target]
                except KeyError:
                    probability = 0.0
                    translation_probabilities[source_with_target] = 0.0
                src_tgt_w_tuple = \
                    source_with_target, probability, t_w_index, s_w_index
                t_w_possible_translations.append(src_tgt_w_tuple)
                # Collect all possible translations
            # Sort by highest probability
            sorted_probabilities = sorted(
                t_w_possible_translations,
                key=operator.itemgetter(1),
                reverse=True
            )
            # Get the one with the highest probability
            word_translation_n_alignment = sorted_probabilities[0]
            word_alignment = \
                str(word_translation_n_alignment[2]) \
                + '-' \
                + str(word_translation_n_alignment[3])
            # The first index shown belongs to tgt_word,
            # second index belongs to src_word
            sentence_alignments.append(word_alignment)
            sentence = " ".join(sentence_alignments)
        sentences_alignments.append(sentence)
    return sentences_alignments
    # List of strings ['0-0 1-2', '0-0 1-3 2-2', '0-0 1-2']


def save_data(tokens, filename):
    """Save tokens into file, separated by comma."""
    with open(filename, "w") as file:
        tokens_string = ",".join(tokens)
        file.write(tokens_string)
    return file
