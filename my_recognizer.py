import warnings
from asl_data import SinglesData
import arpa


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


def recognize_unigram(models: dict, test_set: SinglesData, lm_scaling_factor: int):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :param lm_scaling_factor: int
        multiply the language model probability by int value so it's on a closer scale the the HMM log_ls probability
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

    # read in the language model from the ukn.1.lm arpa file provided
    # note: lm stands for language model

    try:
        language_models = arpa.loadf('./data/ukn.1.lm')
        lm = language_models[0]  # ARPA files may contain several models.
    except:
        print("Problem reading the language model from the ARPA file")
        raise

    # implement the recognizer

    # print("Total Length: {}".format(test_set.num_items))
    for video_num in test_set.sentences_index:
        for word_id in test_set.sentences_index[video_num]:
            log_ls = {}  # dict of log liklihoods of a word
            best_score = float("-inf")  # best log_l thus far
            best_guess = None  # best guess for what the word from the test set could be
            x, lengths = test_set.get_item_Xlengths(word_id)

            for word, model in models.items():

                try:
                    # Assumes a HMM model
                    log_ls[word] = model.score(x, lengths)
                except:
                    # Unable to process word with this model
                    log_ls[word] = float("-inf")
                else:
                    # Remove a trailing digit from word if it has one before passing to language model
                    word_key = ''.join(word[:-1] if word[-1].isdigit() else word)
                    log_ls[word] = log_ls[word] + lm_scaling_factor * lm.log_p(word_key)

                if log_ls[word] > best_score:
                    best_score = log_ls[word]
                    best_guess = word
                    # print("New Best Guess for {}: {}".format(word_id, best_guess))

            probabilities.append(log_ls)
            guesses.append(best_guess)

    return probabilities, guesses


def recognize_bigram(models: dict, test_set: SinglesData, lm_scaling_factor: int):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :param lm_scaling_factor: int
        multiply the language model probability by int value so it's on a closer scale the the HMM log_ls probability
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

    # read in the language model from the ukn.2.lm arpa file provided
    # note: lm stands for language model
    try:
        language_models = arpa.loadf('./data/ukn.2.lm')
        lm = language_models[0]  # ARPA files may contain several models.
    except:
        print("Problem reading the language model from the ARPA file")
        raise

    # implement the recognizer
    prev_sentence = 0  # previous word_id's sentence index
    for video_num in test_set.sentences_index:
        for word_id in test_set.sentences_index[video_num]:
            log_ls = {}  # dict of log liklihoods of a word
            best_score = float("-inf")  # best log_l thus far
            best_guess = None  # best guess for what the word from the test set could be
            x, lengths = test_set.get_item_Xlengths(word_id)
            curr_sentence = video_num

            for word, model in models.items():

                try:
                    # Assumes a HMM model
                    log_ls[word] = model.score(x, lengths)
                except:
                    # Unable to process word with this model
                    log_ls[word] = float("-inf")
                else:
                    # Remove a trailing digit from word if it has one before passing to language model
                    if not guesses or curr_sentence != prev_sentence:  # First guess in a sentence
                        prev_word = '<s>'
                    else:
                        prev_word = guesses[-1]
                        prev_word = ''.join(prev_word[:-1] if prev_word[-1].isdigit() else prev_word)
                    word_key = ''.join(word[:-1] if word[-1].isdigit() else word)
                    word_key = prev_word + ' ' + word_key  # Combine previous word and current word
                    lang_model_probability = lm.log_p(word_key)
                    '''
                    print("----------------------------------")
                    print("Current Sentence: {}".format(curr_sentence))
                    print("Word Key: {}".format(word_key))
                    print("HMM probability: {}".format(log_ls[word]))
                    print("LM probability: {}".format(lang_model_probability*lm_scaling_factor))
                    '''
                    log_ls[word] = log_ls[word] + lm_scaling_factor * lang_model_probability

                if log_ls[word] > best_score:
                    best_score = log_ls[word]
                    best_guess = word
                    # print("New Best Guess for {}: {}".format(word_id, best_guess))

            prev_sentence = curr_sentence

            probabilities.append(log_ls)
            guesses.append(best_guess)

    return probabilities, guesses


def recognize_trigram(models: dict, test_set: SinglesData, lm_scaling_factor: int):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :param lm_scaling_factor: int
        multiply the language model probability by int value so it's on a closer scale the the HMM log_ls probability
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

    # read in the language model from the ukn.3.lm arpa file provided
    # note: lm stands for language model
    try:
        language_models = arpa.loadf('./data/ukn.3.lm')
        lm = language_models[0]  # ARPA files may contain several models.
    except:
        print("Problem reading the language model from the ARPA file")
        raise

    # implement the recognizer
    prev_sentence = -1  # previous word_id's sentence index
    prev_prev_sentence = -2  # two words back, the word_id's sentence index
    for video_num in test_set.sentences_index:
        for word_id in test_set.sentences_index[video_num]:
            log_ls = {}  # dict of log liklihoods of a word
            best_score = float("-inf")  # best log_l thus far
            best_guess = None  # best guess for what the word from the test set could be
            x, lengths = test_set.get_item_Xlengths(word_id)
            curr_sentence = video_num

            for word, model in models.items():

                try:
                    # Assumes a HMM model
                    log_ls[word] = model.score(x, lengths)
                except:
                    # Unable to process word with this model
                    log_ls[word] = float("-inf")
                else:
                    # Remove a trailing digit from word if it has one before passing to language model
                    if not guesses or curr_sentence != prev_sentence:  # First guess in a sentence
                        prev_word = '<s>' + ' ' + '<s>'
                    elif prev_sentence != prev_prev_sentence:  # Second word in a sentence
                        prev_word = guesses[-1]
                        prev_word = '<s>' + ' ' + ''.join(prev_word[:-1] if prev_word[-1].isdigit() else prev_word)
                    else:
                        prev_word = guesses[-1]
                        temp = guesses[-2]
                        prev_word = ''.join(prev_word[:-1] if prev_word[-1].isdigit() else prev_word)
                        temp = ''.join(temp[:-1] if temp[-1].isdigit() else temp)
                        prev_word = temp + ' ' + prev_word
                    word_key = ''.join(word[:-1] if word[-1].isdigit() else word)
                    word_key = prev_word + ' ' + word_key  # Combine previous word and current word
                    log_ls[word] = log_ls[word] + lm_scaling_factor * lm.log_p(word_key)

                if log_ls[word] > best_score:
                    best_score = log_ls[word]
                    best_guess = word
                    # print("New Best Guess for {}: {}".format(word_id, best_guess))

            prev_prev_sentence = prev_sentence
            prev_sentence = curr_sentence

            probabilities.append(log_ls)
            guesses.append(best_guess)

    return probabilities, guesses
