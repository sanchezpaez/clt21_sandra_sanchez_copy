# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 07.04.2022


from learn_alignments import learn_alignments


def test_learn_alignments():
    learn_alignments(
        source_language='src_en_TEST.txt',
        target_language='tgt_fr_TEST.txt',
        model_probabilities_filename='TEST_tiny_probs.txt',
        pairs_filename='TEST_tiny_pairs.txt'
    )
    with open('TEST_tiny_probs_actual.txt', 'r') as actual_file:
        with open('TEST_tiny_probs_expected.txt', 'r') as expected_file:
            assert actual_file.read() == expected_file.read()

    with open('TEST_tiny_pairs_actual.txt', 'r') as actual_file:
        with open('TEST_tiny_pairs_expected.txt', 'r') as expected_file:
            assert actual_file.read() == expected_file.read()
