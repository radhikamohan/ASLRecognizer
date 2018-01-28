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
    for word_id in range(test_set.num_items):
        X, X_lengths = test_set.get_item_Xlengths(word_id)
        best_score = float('-inf')
        best_guess = None
        prob_dict = {}                
        for w,model in models.items():
            logL = float('-inf')
            try:
                logL = model.score(X, X_lengths)
                prob_dict[w] = logL
            except:
                prob_dict[w] = logL
            if logL > best_score:
                best_score = logL
                best_guess = w
        probabilities.append(prob_dict)
        guesses.append(best_guess)
    return probabilities, guesses

