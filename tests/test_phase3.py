# -*- coding: utf-8 -*-
# Modulprojekt CLT
# Authorin: Sandra Sánchez
# Datum: 07.04.2022

from evaluate import evaluate


def test_evaluate():
    recall_value, precision_value, aer_value = evaluate(
        gold_alignments_filename='../gold_standard/goldstandard_en_es.txt',
        gold_source_sentences_filename='1-100-final.en',
        gold_target_sentences_filename='1-100-final.es',
        probabilities_filename='TEST_tiny_probs.txt',
        golden_sents_calculated_alignments_filename='TEST_tiny_golden_calc_alignments.txt'
    )
    assert recall_value == 0.11745334796926454
    assert precision_value == 0.09764918625678119
    assert aer_value == 0.8934060485870104
