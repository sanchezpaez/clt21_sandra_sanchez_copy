# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 07.04.2022


import sys

import os

from shared_functions \
    import read_lines_from_file, clean_corpus_leave_punctuation, \
    get_sentence_pairs, calculate_word_alignments, save_data


def tokenise_already_preprocessed_corpus(preprocessed_corpus):
    """
    Split every list in corpus to access the word level
    :param preprocessed_corpus: list of str of the form
     ['1-0 2-2', '1-0 2-3 3-2', '1-0 2-2']
    :return: list of lists of str of the form
    [['1-0', '2-2'], ['1-0', '2-3', '3-2'], ['1-0', '2-2']]
    """

    sents = [s.split() for s in preprocessed_corpus]
    return sents


def load_data(filename):
    """
    Get text from file and split by ','
    :param filename: file
    :return: list of str
    """
    with open(filename, encoding='utf-8') as file:
        text = file.read()
        words = text.split(',')
    return words


def read_and_preprocess_gold_sentences(gold_sentences_file):
    """Read file with golden sentences and tokenise them."""
    foldername = os.path.dirname(gold_sentences_file)
    full_path = os.path.join('../gold_standard/',
                             foldername, gold_sentences_file)
    with open(full_path, encoding='utf-8') as file:
        text = file.read()  # Text as string
    text = text.strip().split('</s>')  # Split into sentences
    all_sents = []
    for sentence in text:
        if sentence == '':  # Delete empty lines
            pass
        else:
            sentence = sentence.split()
            sentence = sentence[2:]
            sentence = ' '.join(sentence)
            all_sents.append(sentence)
    clean_sents = clean_corpus_leave_punctuation(all_sents)
    return clean_sents


def get_gold_alignments(lines):
    """
    Get gold alignments from lines.
    Group alignments by sentence number and pack them into lists
    :param lines: str
        Example: '1 2 2 S', '1 1 1 S', '2 6 5 S', '2 5 4 S'
    :return: list of lists of tuples
        Example:  [[('93', '11-15', 'S'), ('93', '6-9', 'S'),
         [('94', '8-11', 'S'), ('94', 9-12', 'S')]]
    """
    lists = []
    tuples = []
    previous_index = "1"  # init to first line-index (1-index, not 0-index)
    for line in lines:
        index, match_a, match_b, letter, = line.split()
        alignment_tuple = (f"{match_a}-{match_b}", letter)
        if index == previous_index:
            tuples.append(alignment_tuple)
        else:
            lists.append(tuples)
            tuples = [alignment_tuple]
            previous_index = index
    lists.append(tuples)
    return lists


def add_missing_null_alignments_to_goldstandard(
        gold_sentences_alignments,
        target_sentences_lengths):
    """
    Check all word positions in the target sentence are aligned
    with some word position in the source sentence. If not, align
    them with 'NULL'.
    The format is (
    'source_word position'-'target_word_position',
     'annotation').
    Example of what we use as input:
        [[('4-5', 'S'), ('3-2', 'S'), ('2-2', 'S'), ('1-1', 'S')]]
    Example of what we would return:
        [[('4-5', 'S'), ('3-2', 'S'), ('2-2', 'S'),
         ('1-1', 'S'), ('0-3', 'S'), ('0-4', 'S')]]
    :param gold_sentences_alignments: list of lists of tuples of str.
    :param target_sentences_lengths: list of int.
    :return list of lists of tuples of str.
    """
    for index, gold_sentence_alignments \
            in enumerate(gold_sentences_alignments):
        sentence_length: int = target_sentences_lengths[index]
        used_target_words = [int(alignment.split('-')[-1])
                             for alignment, annotation
                             in gold_sentence_alignments]
        unused_target_words = [word_index for word_index
                               in range(1, sentence_length + 1) if
                               word_index not in used_target_words]
        for unused_target in unused_target_words:
            new_tuple = f"0-{unused_target}", 'S'
            gold_sentence_alignments.append(new_tuple)
    return gold_sentences_alignments


def get_sentences_lengths(sentences_list):
    "Count words in each sentence. Return list of int."
    lengths = []
    for sentence in sentences_list:
        length = len(sentence)
        lengths.append(length)
    return lengths


def precision(gold_alignments, calculated_alignments):
    """
    Return the proportion of possible alignments from all alignments.
    |A ∩ P| / |A|
    Count number of matches in the gold standard disregarding annotation.
    (if 'P' is possible, 'S' is also possible)
    Compute total number of calculated alignments.
    :param gold_alignments: list of lists of tuples of str.
    :param calculated_alignments: list of lists of str.
    :return int: matches/total
    """

    # Find all matching alignments
    matches = get_possible_matches(calculated_alignments, gold_alignments)
    # Compute all alignments
    total = count_all_calculated_alignments(calculated_alignments)

    return matches / total


def recall(gold_alignments, calculated_alignments):
    """
    Return the proportion of S-alignments found by the model.
    |A ∩ S| / |S|
    Count matching alignments (calculated word alignments
    that are annotated as 'S' in the gold standard).
    Count total number of sure alignments in the gold standard.
    :param gold_alignments: list of lists of tuples of str.
    :param calculated_alignments: list of lists of str.
    :return int: matches/sure_total
    """

    # Find matching alignments
    matches = get_sure_matches(calculated_alignments, gold_alignments)
    # Count Sure alignments in gold standard
    sure_total = count_sure_alignments_in_gold_standard(gold_alignments)
    return matches / sure_total


def alignment_error_rate(calculated_alignments, gold_alignments):
    """
    Calculate Alignment Error Rate.
    AER = 1 - (|A∩S| + |A∩P|) / (|A| + |S|)
    :param gold_alignments: list of lists of tuples of str.
    :param calculated_alignments: list of lists of str.
    :return: int between 0 and 1
    """
    # 1 - (matches recall + matches precision) / (all alignments + sure_total)
    all_matches = get_sure_matches(
        calculated_alignments, gold_alignments) \
                  + get_possible_matches(calculated_alignments, gold_alignments)
    totals = count_all_calculated_alignments(calculated_alignments) \
             + count_sure_alignments_in_gold_standard(gold_alignments)
    aer = 1 - (all_matches / totals)
    return aer


def reverse_indexes(alignments_list):
    """
    Reformat the calculated alignments, so they look like the gold standard,
    which is more readable. Afterwards we can more easily compare them.
    We pass in the str 'target_word_position'-'source_word_position'
    and return 'source_word_position'-'target_word_position'.
    """
    reversed_sentences = []
    for sentence_alignment in alignments_list:
        reversed_sentence = []
        for word_alignment in sentence_alignment:  # ['1-0', '2-3', '3-2']
            alignment = word_alignment.split('-')  # ['1', '0']
            reversed_alignment = alignment[1] + '-' + word_alignment[0]
            reversed_sentence.append(reversed_alignment)
        reversed_sentences.append(reversed_sentence)
    return reversed_sentences


def get_possible_matches(calculated_alignments, gold_alignments):
    """
    Get all matching possible alignments.
    Iterate over the calculated alignment list.
    When a word alignment in a sentence is found in its respective
    golden aligned sentence, count it as match.
    :param calculated_alignments: list of lists of str.
    :param gold_alignments: list of lists of tuples of str.
    :return: int
    """
    matches = 0
    pairs = zip(calculated_alignments, gold_alignments)
    for pair in pairs:
        for aligned_word in pair[0]:
            if (aligned_word, 'S') in pair[1]:
                matches += 1
            elif (aligned_word, 'P') in pair[1]:
                matches += 1
            else:
                pass
    return matches


def get_sure_matches(calculated_alignments, gold_alignments):
    """
    Get all matching sure alignments.
    Iterate over the calculated alignment list.
    When a word alignment in a sentence is found in its respective
    golden aligned sentence, and it is annotated as 'Sure', count it as match.
    :param calculated_alignments: list of lists of str.
    :param gold_alignments: list of lists of tuples of str.
    :return: int
    """
    matches = 0
    pairs = zip(calculated_alignments, gold_alignments)
    for pair in pairs:
        for aligned_word in pair[0]:
            if (aligned_word, 'S') in pair[1]:
                matches += 1
            else:
                pass
    return matches


def count_sure_alignments_in_gold_standard(gold_alignments):
    """
    Count golden word alignments annotated as sure ('S').
    Example: [[('4-5', 'S'), ('3-2', 'S')], [('2-2', 'P')]] would return 2.
    :param gold_alignments: list of lists of tuples of str.
    :rtype: int
    """
    sure_total = 0

    for gold_sent in gold_alignments:
        for word_alignment in gold_sent:
            if word_alignment[1] == 'S':
                sure_total += 1
            else:
                pass
    return sure_total


def count_all_calculated_alignments(calculated_alignments):
    """Compute all calculated alignments. Return int."""
    total = 0
    for sentence_aligned in calculated_alignments:
        for alignment in sentence_aligned:
            total += 1
    return total


def evaluate(
        gold_alignments_filename: str,
        gold_source_sentences_filename: str,
        gold_target_sentences_filename,
        probabilities_filename: str,
        golden_sents_calculated_alignments_filename: str
):
    """
    Use trained model to get word alignments from
    a corpus. Compare calculated alignments with annotated
    alignments reference file and determine recall,
    precision and AER.
    :param gold_alignments_filename: file with
        alignments and annotations.
    :param gold_source_sentences_filename: file with
        golden sentences of source language.
    :param gold_target_sentences_filename: file with
        golden sentences of target language.
    :param probabilities_filename: file with modelled
        probabilities.
    :param golden_sents_calculated_alignments_filename:
        file with calculated alignments
    :return: recall, precision and AER values.
    """
    print("________________PHASE 3: EVALUATE_____________________")
    gold_lines = read_lines_from_file(gold_alignments_filename)
    gold_alignments = get_gold_alignments(gold_lines)
    gold_clean_sents_en = read_and_preprocess_gold_sentences(
        gold_source_sentences_filename)
    gold_clean_sents_es = read_and_preprocess_gold_sentences(
        gold_target_sentences_filename)
    gold_sentence_pairs = get_sentence_pairs(
        gold_clean_sents_en, gold_clean_sents_es)
    target_sentences_lengths = get_sentences_lengths(gold_clean_sents_es)
    complete_gold_alignments = add_missing_null_alignments_to_goldstandard(
        gold_alignments, target_sentences_lengths)
    golden_calculated_sentences_alignments = calculate_word_alignments(
        probabilities_filename, gold_sentence_pairs)
    save_data(golden_calculated_sentences_alignments,
              golden_sents_calculated_alignments_filename)
    loaded_golden_calculated_sentences_alignments = load_data(
        golden_sents_calculated_alignments_filename)
    tokenised_alignments = tokenise_already_preprocessed_corpus(
        loaded_golden_calculated_sentences_alignments)
    reversed_alignments = reverse_indexes(tokenised_alignments)
    recall_value = recall(complete_gold_alignments, reversed_alignments)
    print(recall_value)
    precision_value = precision(
        complete_gold_alignments, reversed_alignments)
    print(precision_value)
    aer_value = alignment_error_rate(
        reversed_alignments, complete_gold_alignments)
    print(aer_value)
    return recall_value, precision_value, aer_value


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(f"6 arguments are needed, found {len(sys.argv)}")
    else:
        gold_alignments_filename = sys.argv[1]
        gold_source_sentences_filename = sys.argv[2]
        gold_target_sentences_filename = sys.argv[3]
        probabilities_filename = sys.argv[4]
        golden_sents_calculated_alignments_filename = sys.argv[5]
        evaluate(
            gold_alignments_filename=sys.argv[1],
            gold_source_sentences_filename=sys.argv[2],
            gold_target_sentences_filename=sys.argv[3],
            probabilities_filename=sys.argv[4],
            golden_sents_calculated_alignments_filename=sys.argv[5]
        )
