# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 07.04.2022

import sys

from shared_functions\
    import calculate_word_alignments, save_data


def align_words(modelled_probabilities: str,
                calculated_alignments_filename: str,
                sentence_pairs_filename: str):
    """
    Get words translations and alignments based on the
    translation probabilities calculated by the EM algorithm.
    Save them into file.
    :param modelled_probabilities: File with translation probs.
    :param calculated_alignments_filename: File to save
        alignments into.
    :param sentence_pairs_filename: file with sentence pairs.
    :return: None
    """
    print("________________PHASE 2: ALIGN WORDS__________________")

    tiny_sentence_pairs = get_sentence_pairs_from_file(
        sentence_pairs_filename)
    alignments = calculate_word_alignments(
        modelled_probabilities, tiny_sentence_pairs)
    save_data(alignments, calculated_alignments_filename)


def get_sentence_pairs_from_file(sentence_pairs_filename: str):
    """
    Convert str into list of lists (sentence pairs).
    Split and group every two items [source_sentence, target_sentence]
    :param sentence_pairs_filename: file to read from
    :return: list of lists of str.
    """
    sentence_pairs = []
    with open(sentence_pairs_filename, 'r') as file:
        for index, line in enumerate(file.readlines()):
            tokenise = line.split()
            is_source: bool = index % 2 == 0
            if is_source:
                sentence_pairs.append([tokenise])
            else:
                sentence_pairs[-1].append(tokenise)
    return sentence_pairs


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(f"4 arguments are needed, found {len(sys.argv)}")
    else:
        modelled_probabilities = sys.argv[1]
        calculated_alignments_filename = sys.argv[2]
        sentence_pairs_filename = sys.argv[3]
        align_words(
            modelled_probabilities,
            calculated_alignments_filename,
            sentence_pairs_filename
        )
