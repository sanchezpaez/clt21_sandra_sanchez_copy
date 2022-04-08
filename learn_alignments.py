# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 07.04.2022

import sys

import nltk

nltk.download('punkt')

from tqdm import tqdm

from shared_functions \
    import read_lines_from_file, clean_corpus_leave_punctuation, \
    get_sentence_pairs


def select_smaller_corpus_from_corpus(corpus, minimize=False):
    """
    Set split size to select a percentage of sentences
    from original corpus to get a smaller corpus.
    :param corpus: list of sentences (str)
    :param minimize: bool, when True selects a minimal portion of a corpus
    :return: list of sentences (str)
    """
    split_size = int(len(corpus) * 0.5)
    if minimize:
        split_size = 3000
    partial_corpus = corpus[:split_size]
    return partial_corpus


def get_unique_words(sentences):
    """
    Get unique words from list of sentences.
    Sort them in alphabetical order.
    :param sentences: list of lists of str
    :return: list of str
    """
    vocab = set().union(*sentences)
    vocab = sorted(list(vocab))
    return vocab


def expectation_maximization_algorithm(
        source_words, target_words, parallel_corpus, filename):
    """
    Run the expectation-maximization algorithm on a parallel corpus
    in order to find the most likely word translations that
    we need to calculate world alignments.
    :param source_words: list of str.
        Example: ['NULL', 'blue', 'flower', 'house', 'the']
    :param target_words: list of str.
        Example: ['bleu', 'fleur', 'la', 'maison']
    :param parallel_corpus: list of tuples
        ([source_sentence], [target_sentence])
    :param filename: file into which we save the probabilities as str
    :return: dict with items of the form (s_w, t_w) : float,
    where the float represents the probability
    Example: ('house', 'maison'): 0.4862535128673125
    """
    t_probs = initialise(source_words, target_words)
    print("Probabilities initialised")
    s_total = {}
    number_iterations = 3
    for iteration in tqdm(range(number_iterations)):
        count = {}
        total = {}

        for s_w in source_words:
            for t_w in target_words:
                total[t_w] = 0.0
                count[(s_w, t_w)] = 0.0
        print('1/3 for loop done')

        for pair in parallel_corpus:
            for s_w in pair[0]:  # // Normalization
                s_total[s_w] = 0.0
                for t_w in pair[1]:
                    s_total[s_w] += t_probs[(s_w, t_w)]
            # // E-Step
            for s_w in pair[0]:
                for t_w in pair[1]:
                    count[(s_w, t_w)] += t_probs[(s_w, t_w)] / s_total[s_w]
                    total[t_w] += t_probs[(s_w, t_w)] / s_total[s_w]
        print('2/3 for loop done')
        # // M-Step
        for t_w in target_words:
            for s_w in source_words:
                t_probs[(s_w, t_w)] = count[(s_w, t_w)] / total[t_w]
        print('3/3 for loop done')
        print(f"Finished iteration number {iteration + 1}/3")

    save_probs_into_file_tab(t_probs, filename)  # translation prob. t(e|f)


def save_probs_into_file_tab(probabilities_dict, filename):
    """Save probabilities items into file, separated by tab and new line."""
    all_probs = []
    for prob in probabilities_dict:
        string_prob = prob[0] + "\t" + prob[1] + "\t" + str(probabilities_dict[prob])
        all_probs.append(string_prob)
    with open(filename, "w") as file:
        tokens_string = '\n'.join(all_probs)
        file.write(tokens_string)


def initialise(source_language_set, target_language_set):
    """
    Assign initial value to all (source_word, target_word) combinations.
    Every word combination is equally probable.
    :param source_language_set: list of str
    :param target_language_set: list of str
    :return: dict of items that are tuple: float
    """
    probs = {}
    initial_prob = 1 / len(target_language_set)
    for f_w in target_language_set:
        for e_w in source_language_set:
            pair = (e_w, f_w)
            probs[pair] = initial_prob
    return probs


def learn_alignments(
        source_language: str, target_language: str,
        model_probabilities_filename: str, pairs_filename: str):
    """
    Phase 1: calculate translation probabilities by calling the
    expectation maximization algorithm
    :param source_language: file with source sentences
    :param target_language: file with target sentences
    :param model_probabilities_filename: file to save calculated probs
    :param pairs_filename: file to save sentence pairs
    :return: None
    """
    print("________________PHASE 1: LEARN ALIGNMENTS_______________")
    source_corpus_raw = read_lines_from_file(source_language)
    target_corpus_raw = read_lines_from_file(target_language)
    # This step is important in larger corpora
    partial_corpus_en = select_smaller_corpus_from_corpus(source_corpus_raw,
                                                          minimize=True)
    partial_corpus_es = select_smaller_corpus_from_corpus(
        target_corpus_raw, minimize=True)
    print("Preprocessing corpora.")
    preprocessed_source_sents = \
        clean_corpus_leave_punctuation(partial_corpus_en, append_null=True)
    preprocessed_target_sents = \
        clean_corpus_leave_punctuation(partial_corpus_es)
    print("Getting sentence pairs.")
    tiny_sentence_pairs = get_sentence_pairs(
        preprocessed_source_sents, preprocessed_target_sents)
    with open(pairs_filename, 'w') as file:
        lines = ''
        for source, target in tiny_sentence_pairs:
            lines += f"{' '.join(source)}\n{' '.join(target)}\n"
        file.write(lines)
    print("Getting vocabularies.")
    source_words = get_unique_words(preprocessed_source_sents)
    foreign_words = get_unique_words(preprocessed_target_sents)
    print("Running expectation maximization algorithm to get translation probabilities.")
    expectation_maximization_algorithm(
        source_words, foreign_words, tiny_sentence_pairs, model_probabilities_filename)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit(f"5 arguments are needed, found {len(sys.argv)}")
    else:
        source_language = sys.argv[1]
        target_language = sys.argv[2]
        model_probabilities_filename = sys.argv[3]
        pairs_filename = sys.argv[4]
        learn_alignments(
            source_language,
            target_language,
            model_probabilities_filename,
            pairs_filename
        )
