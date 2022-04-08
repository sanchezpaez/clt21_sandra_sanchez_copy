# Repository für das CLT-Modulprojekt, WiSe 21/22 


Installation and functionality description
=======================

1. Intro
-------

This directory contains a codification of the IBM 1 model to calculate lexical translations.

The program has three main parts (`learn_alignments.py`, `align_words.py` and `evaluate.py`) that are run independently from each other and are called from the command line.


The directory contains:

* `learn_alignments.py`
* `align_words.py`
* `evaluate.py`
* gold_standard(folder):
* tests(folder):
* output(folder): 



2. Installation
-------

1) Clone the repository.

2) Using your terminal navigate through your computer to find the directory were you cloned the repository. Then from Terminal (look for 'Terminal' on Spotlight), or CMD for Windows,  set your working directory to that of your folder (for example: cd Desktop/clt21_sandra_sanchez).

3) Download the es-en (Spanish-English) corpus from the europarl website (https://www.statmt.org/europarl/) and save it into the directory where you saved the repository. Not available on 07.04.2022!!!

Your folder structure should look like this:

```
clt_sand
  ↳
   …
   es-en
      ↳
      europarl-v7.es-en.en
      europarl-v7.es-en.es
  
```

4) Required packages:

If you don't have pip installed follow the installing instructions here: https://pip.pypa.io/en/stable/installation/

Install the required package tqdm by typing the following on your terminal:

```
pip install tqdm
```

If you don't have nltk installed follow the installing instructions here:

https://www.nltk.org/install.html

If you have problems downloading the nltk.punkt package, add this to the top of your learn_alignments.py file:

```
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
```

5) You should be able to run the script now. Check first how you can run python on your computer (it can be 'python' or 'python3').

First phase: learn word alignments

Go to the command line and type the following:

```
python learn_alignments.py es-en/europarl-v7.es-en.en es-en/europarl-v7.es-en.es translation_probabilities_model.txt sentence_pairs.txt
```

Second phase: align words

Go to the command line and type the following:

```
python align_words.py translation_probabilities_model.txt calculated_alignments.txt sentence_pairs.txt
```

Third phase: evaluate

Go to the command line and type the following:

```
python evaluate.py goldstandard_en_es.txt 1-100-final.en 1-100-final.es translation_probabilities_model.txt golden_calculated_alignments.txt
```
You will see printed the values for recall, precision and AER.

6) To run the tests:

install pytest:

```
pip install pytest
```

The tests are in the tests folder, and are the following:

test_helper_functions.py
test_phase1.py
test_phase2.py
test_phase3.py

There is one test for each one of the 3 phases, and another one to test all the helper functions.

There were inconsistencies while running them in the temrinal but they are well coded and work perfectly on pycharm, so just run every file directly from pycharm.


3. Contact information
-------

If you have any questions or problems during they installation process, feel free to email sandrasanchezp@hotmail.com
