# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 07.04.2022

from evaluate\
    import get_gold_alignments, read_and_preprocess_gold_sentences, \
    reverse_indexes, add_missing_null_alignments_to_goldstandard, \
    get_possible_matches, get_sure_matches, \
    count_sure_alignments_in_gold_standard, \
    count_all_calculated_alignments, precision, \
    recall, alignment_error_rate
from learn_alignments \
    import expectation_maximization_algorithm, initialise, \
    save_probs_into_file_tab
from shared_functions \
    import tokenize_not_remove_punctuation, \
    clean_corpus_leave_punctuation, read_lines_from_file


class TestsGolden:

    def test_get_golden_sentences_alignments_from_empty_lines(self):
        actual = get_gold_alignments(lines=[])
        expected = [[]]
        assert actual == expected

    def test_get_golden_sentences_alignments_from_one_line(self):
        actual = get_gold_alignments(lines=["1 4 5 S"])
        expected = [[('4-5', 'S')]]
        assert actual == expected

    def test_get_golden_sentences_alignments_from_several_lines(self):
        actual = get_gold_alignments(
            lines=["1 4 5 S", "1 3 2 S", "1 2 2 S", "1 1 1 S"]
        )
        expected = [[('4-5', 'S'), ('3-2', 'S'), ('2-2', 'S'), ('1-1', 'S')]]
        assert actual == expected

    def test_get_golden_sentences_alignments_from_several_sentences(self):
        actual = get_gold_alignments(
            lines=["1 4 5 S", "1 3 2 S", "2 2 2 S", "2 1 1 S"]
        )
        expected = [
            [
                ('4-5', 'S'), ('3-2', 'S')
            ],
            [
                ('2-2', 'S'), ('1-1', 'S')
            ]
        ]
        assert actual == expected


def test_tokenize_not_remove_punctuation():
    actual = tokenize_not_remove_punctuation(
        'The house is blue.',
        append_null=True
    )
    expected = ['NULL', 'the', 'house', 'is', 'blue', '.']
    assert actual == expected


def test_clean_corpus_leave_punctuation():
    actual = clean_corpus_leave_punctuation(
        ['The house is blue.', 'the houses.'],
        append_null=True
    )
    expected = [
        ['NULL', 'the', 'house', 'is', 'blue', '.'],
        ['NULL', 'the', 'houses', '.']
    ]
    assert actual == expected


def test_read_lines_from_file():
    actual = read_lines_from_file('../gold_standard/goldstandard_en_es.txt')
    assert actual[0] == '1 4 5 S'
    assert actual[-1] == '100 2 2 P'


def test_expectation_maximization_algorithm():
    expectation_maximization_algorithm(
        source_words=['NULL', 'the', 'blue', 'house', 'flower'],
        target_words=['la', 'maison', 'bleu', 'fleur'],
        parallel_corpus=[(
            ['NULL', 'the', 'house'],
            ['la', 'maison'],
        ),
            (
                ['NULL', 'the', 'blue', 'house'],
                ['la', 'maison', 'bleu'],
            ),
            (
                ['NULL', 'the', 'flower'],
                ['la', 'fleur'],
            ),
        ],
        filename='em_TEST.txt'
    )
    with open('em_TEST.txt', 'r') as file:
        actual = file.readlines()[0]
        expected = 'NULL\tla\t0.3967162216116505\n'
        assert actual == expected


def test_initialise_0():
    actual = initialise(
        source_language_set=['NULL', 'the', 'blue', 'house', 'flower'],
        target_language_set=['la', 'maison', 'bleu', 'fleur'],
    )
    expected = {
        ('NULL', 'la'): 0.25,
        ('the', 'la'): 0.25,
        ('blue', 'la'): 0.25,
        ('house', 'la'): 0.25,
        ('flower', 'la'): 0.25,
        ('NULL', 'maison'): 0.25,
        ('the', 'maison'): 0.25,
        ('blue', 'maison'): 0.25,
        ('house', 'maison'): 0.25,
        ('flower', 'maison'): 0.25,
        ('NULL', 'bleu'): 0.25,
        ('the', 'bleu'): 0.25,
        ('blue', 'bleu'): 0.25,
        ('house', 'bleu'): 0.25,
        ('flower', 'bleu'): 0.25,
        ('NULL', 'fleur'): 0.25,
        ('the', 'fleur'): 0.25,
        ('blue', 'fleur'): 0.25,
        ('house', 'fleur'): 0.25,
        ('flower', 'fleur'): 0.25,
    }
    assert actual == expected


def test_initialise_1():
    actual = initialise(
        source_language_set=['NULL', 'the', 'blue'],
        target_language_set=['la', 'bleu'],
    )
    expected = {
        ('NULL', 'la'): 0.5,
        ('the', 'la'): 0.5,
        ('blue', 'la'): 0.5,
        ('NULL', 'bleu'): 0.5,
        ('the', 'bleu'): 0.5,
        ('blue', 'bleu'): 0.5,
    }
    assert actual == expected


def test_read_and_preprocess_gold_sentences():
    actual = read_and_preprocess_gold_sentences(
        '1-100-final.en')
    assert len(actual) == 100
    assert actual[0] == ['resumption', 'of', 'the', 'session']
    assert actual[-1] == ['we', 'must', 'show', 'they', 'are', 'wrong', '.']
    assert actual[20] == ['there', 'are', 'not', ',', 'therefore',
                          ',', 'any', 'amendments', 'to', 'the', 'agenda',
                          'for', 'friday', '.']


def test_reverse_indexes():
    actual = reverse_indexes(
        [['0-0', '1-2'], ['0-0', '1-3', '2-2'], ['0-0', '1-2']]
    )
    expected = [['0-0', '2-1'], ['0-0', '3-1', '2-2'], ['0-0', '2-1']]
    assert actual == expected


def test_add_missing_null_alignments_to_goldstandard():
    actual = add_missing_null_alignments_to_goldstandard(
        gold_sentences_alignments=[
            [('4-5', 'S'), ('3-2', 'S'), ('2-2', 'S'), ('1-1', 'S')],
            [('4-5', 'S'), ('3-2', 'S'), ('2-2', 'S'), ('1-1', 'S')]],
        target_sentences_lengths=[5, 5]
    )

    expected = [
        [('4-5', 'S'), ('3-2', 'S'), ('2-2', 'S'),
         ('1-1', 'S'), ('0-3', 'S'), ('0-4', 'S')],
        [('4-5', 'S'), ('3-2', 'S'), ('2-2', 'S'),
         ('1-1', 'S'), ('0-3', 'S'), ('0-4', 'S')]
    ]
    assert actual == expected


def test_get_possible_matches():
    actual = get_possible_matches(
        calculated_alignments=[
            ['0-1', '2-2', '0-3'], ['1-4', '0-5']
        ],
        gold_alignments=[
            [('4-5', 'S'), ('2-2', 'S')],
            [('2-2', 'S'), ('1-4', 'P')]
        ]
    )
    expected = 2
    assert actual == expected


def test_get_sure_matches():
    actual = get_sure_matches(
        calculated_alignments=[
            ['0-1', '2-2', '0-3'], ['1-4', '0-5']
        ],
        gold_alignments=[
            [('4-5', 'S'), ('2-2', 'S')],
            [('2-2', 'S'), ('1-4', 'P')]
        ]
    )
    expected = 1
    assert actual == expected


def test_count_sure_alignments_in_gold_standard():
    actual = count_sure_alignments_in_gold_standard(
        gold_alignments=[[('4-5', 'S'), ('3-2', 'S')], [('2-2', 'P')]]
    )
    expected = 2
    assert actual == expected


def test_count_all_calculated_alignments():
    actual = count_all_calculated_alignments(
        calculated_alignments=[
            ['0-1', '2-2', '0-3'], ['1-4', '0-5']
        ]
    )
    expected = 5
    assert actual == expected


def test_precision():
    actual = precision(
        gold_alignments=[[('4-5', 'S'), ('3-2', 'S')], [('2-2', 'P')]],
        calculated_alignments=[['0-1', '3-2', '0-3'], ['2-2', '0-5']]
    )
    expected = 0.4
    assert actual == expected


def test_recall():
    actual = recall(
        gold_alignments=[[('4-5', 'S'), ('3-2', 'S')], [('2-2', 'P')]],
        calculated_alignments=[['0-1', '3-2', '0-3'], ['2-2', '0-5']]
    )
    expected = 0.5
    assert actual == expected


def test_alignment_error_rate():
    actual = alignment_error_rate(
        calculated_alignments=[['0-1', '3-2', '0-3'], ['2-2', '0-5']],
        gold_alignments=[[('4-5', 'S'), ('3-2', 'S')], [('2-2', 'P')]]
    )
    expected = 0.5714285714285714
    assert actual == expected


def test_save_tab():
    save_probs_into_file_tab(
        probabilities_dict={
            ('NULL', 'la'): 0.3967162216116505,
            ('the', 'la'): 0.3967162216116505,
            ('blue', 'la'): 0.023004922711340352,
        },
        filename='test_tab.txt',
    )
    with open('test_tab.txt', "r") as file:
        assert file.read() == 'NULL\tla\t0.3967162216116505\nthe\tla\t0.3967162216116505\nblue\tla\t0.023004922711340352'
