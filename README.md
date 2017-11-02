# AIND_ASL_Recognizer
American Sign Language Recognizer Implementation. Project for the Udacity AI Nanodegree program.

## Overview
The objective of this project is to identify American Sign Language (ASL) words and phrases from a pre-recorded database. Hidden Markov Models (HMMs) and n-gram language models are used to recognize words based on a collection of features extracted from the data set. The dataset can be found in the `asl_recognizer/data/` directory and was derived from 
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php). 
The handpositions (`hand_condensed.csv`) are pulled directly from the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml).The exctracted features used as HMM input include the position of the speaker's hands relative to the speaker's nose and the change in the speaker's hand locations frame-to-frame.

## Getting Started
The project requires Python 3 and the following libraries: 
* [Numpy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/0.17/install.html)
* [Pandas](http://pandas.pydata.org/)
* [Jupyter](http://ipython.org/notebook.html)
* [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)
* [matplotlib](http://matplotlib.org/)
* [arpa](https://pypi.python.org/pypi/arpa/0.1.0b1)

To start the project run:
`jupyter notebook asl_recognizer.ipynb`
