# Bayes Theorem Spam Filter
This is a simple application of Bayes' Theorem for a spam filter classifier

## Bayes' Theorem

The Bayes' Theoreom is at the base of conditional probability and is defined as:

![BayesTheoremFormula]()

Where:
* ![PosteriorProbability](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28h%7CD%29) is the **posterior probability**: what we are trying to estimate.
* ![Likelihood](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28h%7CD%29%5Cfrac%7B%5Ctextbf%7BP%7D%28D%7Ch%29*%5Ctextbf%7BP%7D%28h%29%7D%7B%5Ctextbf%7BP%7D%28D%29%7D) is the **likelihood**: a conditional probability that can be found from data we can obtain from some process.
* ![PriorProbability](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28H%29) is the **prior probability**: the probability we already know and is being updated in the posterior probability.
* ![Evidence](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28H%29) is the **evidence**: the new piece of data that we are taking in consideration to update the posterior probability.

 Note that the notations 'h' and 'D' could be anything but in the context of machine learning they are usually chosen to indicate hypothesis and Data.
 
 For the spam filter classifier the Bayes' Theorem becomes:
 
 ![FormulaForClassifier](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28isSpam%7Cword%29%20%3D%20%5Ctextbf%7BP%7D%28word%7CisSpam%29*%5Ctextbf%7BP%7D%28isSpam%29)
 
 Here our hypothesis is the occurrance of a word in spams and hams ( ![isSpam](http://latex.codecogs.com/png.latex?isSpam) ), and the data is each word in a given email ( ![word)(http://latex.codecogs.com/png.latex?word) ).
 We are trying to find the probability of the hypothesis given the data ( ![hgd](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28isSpam%7Cword%29) ) multiplying the probability of the data given the hypothesis ( ![dgh](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28word%7CisSpam%29) ) by the probability of the hypothesis ( ![h](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28isSpam%29) ).
 The probability of the data given the hypothesis ( ![dgh](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28word%7CisSpam%29) ) is the bit we can 'train' with our dataset in the classifier and the probability of the hypothesis ( ![h](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28isSpam%29) ) is the one we assume, for both cases spam and ham, and compare the resulting probabilities to give a final classification for a new message.
 
 Note that the denominator is being ignored here. It would have been the probability of a word be contained in an email regardless of it being a spam or ham ( ![evidence](http://latex.codecogs.com/png.latex?%5Ctextbf%7BP%7D%28word%29) ).  This is not taken in consideration because it is not relevant and more importantly **It is just a normalization constant, which doesn't depend on the parameter**.


## Overview

The sample dataset provided is from [this](https://www.kaggle.com/uciml/sms-spam-collection-dataset) Kaggle dataset.
The classifier is very basic and can be improved greatly. It is meant to demonstrate how the Bayes' Theorem is applicable to Machine Learning.

## Dependencies

* numpy
* pandas
* sklearn

Install these using [pip](https://pip.pypa.io/en/stable/)

## Usage

Type `python sample_code.py` to run the code.


## Credits

I have re-adapted the code of [AlanBuzdar](https://github.com/alanbuzdar).
