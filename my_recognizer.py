import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # DONE implement the recognizer

    # print("Total Length: {}".format(test_set.num_items))
    for word_id in range(test_set.num_items):
        log_ls = {}  # dict of log liklihoods of a word
        best_score = float("-inf")  # best log_l thus far
        best_guess = None  # best guess for what the word from the test set could be
        x, lengths = test_set.get_item_Xlengths(word_id)

        for word, model in models.items():

            try:
                # Assumes a HMM model
                log_ls[word] = model.score(x, lengths)
            except:
                try:
                    # Assumes a bounded Selector.select method
                    log_ls[word] = model().score(x, lengths)
                except:
                    # Unable to process word with this model
                    log_ls[word] = float("-inf")

            if log_ls[word] > best_score:
                best_score = log_ls[word]
                best_guess = word
                # print("New Best Guess for {}: {}".format(word_id, best_guess))

        probabilities.append(log_ls)
        guesses.append(best_guess)

    return probabilities, guesses
