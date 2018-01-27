import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        https://discussions.udacity.com/t/how-to-start-coding-the-selectors/476905/10
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        selected_model = self.base_model(self.n_constant)
        model_score = float('inf')
        for no_of_comps in range(self.min_n_components, self.max_n_components+1):
            try:
                best_model = self.base_model(no_of_comps)
                logL = best_model.score(self.X, self.lengths)
                no_of_features = self.X.shape[1]
                p = (no_of_comps * no_of_comps-1) + 2 * no_of_features * no_of_comps
                logN = np.log(self.X.shape[0])
                bic = -2 * logL + p * logN
                if bic < model_score:
                    model_score = bic
                    selected_model = best_model
            except:
                continue
        return selected_model 

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    https://discussions.udacity.com/t/how-to-start-coding-the-selectors/476905/10
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        selected_model = None
        model_score = 0
        #list of tuples, 0 - # of components, 1 - word, 2 - score, 3 - model
        comp_model_score= []
        for n_states in range(self.min_n_components, self.max_n_components+1):    
            for w in self.words.keys():
                try:
                    w_X, w_lengths = self.hwords[w]
                    comp_model = self.base_model(n_states)
                    comp_score = comp_model.score(w_X,w_lengths)
                    comp_model_score.append((n_states,w,comp_score,comp_model)) 
                except:
                    continue    
        for n_states in range(self.min_n_components, self.max_n_components+1):
                same_comp_tuples = [t for t in comp_model_score if t[0] == n_states]
                this_word_tuple_list = [t for t in same_comp_tuples if t[1] == self.this_word]
                if(len(this_word_tuple_list)>0):
                    this_word_tuple = this_word_tuple_list[0]
                    this_word_score = this_word_tuple[2]
                    avg_other_words_score = np.average([w[0] for w in same_comp_tuples if w[1] != self.this_word])
                    dic = this_word_score - avg_other_words_score
                    if dic > model_score:
                        dic = model_score
                        selected_model = this_word_tuple[3]
        if selected_model is None:
            return self.base_model(self.n_constant)
        else:
            return selected_model
  
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    https://discussions.udacity.com/t/how-to-start-coding-the-selectors/476905/10
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        selected_model = None
        model_score = float('-inf')
        scores = []
        for n_states in range(self.min_n_components, self.max_n_components+1):
            avg_score = float('-inf')
            if len(self.sequences) < 3:
                break
            split_method = KFold(3, shuffle=False,random_state=self.random_state)
            for cv_train_index, cv_test_index in split_method.split(self.sequences):
                X_train, X_train_lengths = combine_sequences(cv_train_index, self.sequences)
                X_test, X_test_lengths = combine_sequences(cv_test_index, self.sequences)
                model = self.base_model(n_states).fit(X_train, X_train_lengths)
                scores.append(model.score(X_test, X_test_lengths))
            if(len(scores)>0):
                avg_score = np.average(scores)
            if(avg_score > model_score):
                model_score = avg_score
                selected_model = self.base_model(n_states)
        if(selected_model is None):
            return self.base_model(self.n_constant)
        else:
            return selected_model
    