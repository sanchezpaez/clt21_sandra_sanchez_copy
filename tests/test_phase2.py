# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 07.04.2022


from align_words import align_words


def test_align_words():
    align_words(
        modelled_probabilities='trans_prob_TEST.txt',
        calculated_alignments_filename='align_TEST_actual.txt',
        sentence_pairs_filename='sentence_pairs_TEST.txt'
    )
    with open('align_TEST_actual.txt', 'r') as actual_file:
        with open('align_TEST_expected.txt', 'r') as expected_file:
            assert actual_file.read() == expected_file.read()
