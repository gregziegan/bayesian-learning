# Probabilistic Learning

An implementation of Naive Bayes and Logistic Regression classifiers.

### Note: Currently uses much too specific parser for class at CWRU. Will soon be made general purpose. But don't use ours, use [scikit-learn's](http://scikit-learn.org/stable/modules/naive_bayes.html)!

## Installation

    python setup.py install
    
Alternative:

If you would like to work in a virtualenv or install to your global python directory:

    pip install requirements.txt
    
## Execution

    python src/nbayes.py <data_folder_name> <m_estimate>
    python src/logreg.py <data_folder_name> <m_estimate>
