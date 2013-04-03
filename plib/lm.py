"""
Utilities for working with language models.
"""

import kenlm
import math
import numpy as np
import sys
from collections import deque

def ppl(lls, base=10):
    """
    Compute the perplexity using the iterable of log likelihoods.
    """
    length = len(list(lls))
    return base ** (- sum(lls) / length)

def parse_srilm_ppl(stream):
    """
    Parse a perplexity file produced by the SRILM toolkit at debugging
    level 2. Log probabilities are given in base 10.

    Returns

      (1) List of sentence_scores, where a sentence score is a list of
      tuples that can be destructured using (word, lmword, ll). Word
      is the word from the original sentence, lmword is the word that
      was seen by the LM (could be <unk> for example), and ll is the
      log-likelihood (base 10).

    """
    def parse_word_score(s, WORD=1, LOGP=-2):
        fields = s.strip().split()
        return fields[WORD], float(fields[LOGP])

    def parse_sentence_score(s):
        lines = s.strip().split('\n')
        sentence = lines[0].split() + ['</s>']
        word_probs = (parse_word_score(l) for l in lines if l.startswith('\t'))
        return [(w,wlm,ll) for w, (wlm, ll) in zip(sentence, word_probs)]

    chunks = stream.read().strip().split('\n\n')
    sentence_scores = [parse_sentence_score(s) for s in chunks[:-1]]
    return sentence_scores

def interpolate_models(*likelihood_lists, **kwargs):
    """
    Use likelihood predictions from a collection of models to compute
    the optimal mixture weights.

    The likelihood lists passed as input should all be of the same
    length (corresponding to predictions of the same word with the
    same history) and should be probabilities (i.e. not log
    probabilities).

    Returns a tuple of mixture weights that sum to 1.

    """
    tol = kwargs.get('tol', 0.001)
    verbose = kwargs.get('verbose', False)

    likelihoods = np.array(zip(*likelihood_lists))
    nwords, nmodels = likelihoods.shape
    weights = np.ones(nmodels) / nmodels
    converged = False

    while not converged:
        old_ll = np.log10(likelihoods.dot(weights)).sum()
        partition = likelihoods.dot(weights)
        posterior = (likelihoods.dot(np.diag(weights)).T / partition).T
        weights = posterior.sum(axis=0) / nwords
        new_ll = np.log10(likelihoods.dot(weights)).sum()

        if verbose:
            sys.stderr.write('interpolate_models: LL= {:.4f} ( delta= {:.4f} )\n'.\
                                 format(new_ll, abs(new_ll-old_ll)))
        if abs(new_ll-old_ll) < tol:
            converged = True

    if verbose:
        sys.stderr.write('interpolate_models: weights= {}\n'.\
                             format(tuple(weights)))
    return tuple(weights)

class ArpaLanguageModel(object):
    def __init__(self, arpa_file):
        self._path = arpa_file
        self._arpa_lm = kenlm.LanguageModel(arpa_file)
        self._total_likelihood = 0.0

    def log_prob(self, sentence, full=False):
        """
        Compute the log probability (base 10) of the
        sentence. Sentence can either be a string containing
        space-separated tokens, or a sequence of tokens.

        If full=True, the function returns a list of log
        probabilities, one for each token plus an additional
        likelihood for the end of sentence. If full=False a single log
        likelihood is returned.

        """
        if full:
            score_fn = lambda s: [x[0] for x in self._arpa_lm.full_scores(s)]
        else:
            score_fn = self._arpa_lm.score

        if isinstance(sentence, str):
            self._total_likelihood += self._arpa_lm.score(sentence)
            return score_fn(sentence)
        else:
            self._total_likelihood += self._arpa_lm.score(' '.join(sentence))
            return score_fn(' '.join(sentence))

    def total_likelihood(self):
        return self._total_likelihood

    def reset(self):
        self._total_likelihood = 0.0

    def __repr__(self):
        return 'ArpaLanguageModel(path={})'.format(self._path)

class CacheModel(object):
    def __init__(self, size):
        self.size = size
        self.cache = deque(maxlen=size)

    def log_prob(self, w):
        if len(self.cache) == 0:
            self.cache.append(w)
            return -inf
        elif self.cache.count(w) == 0:
            self.cache.append(w)
            return -inf
        else:
            log_prob = math.log10(float(self.cache.count(w))
                                  / len(self.cache))
            self.cache.append(w)
            return log_prob

    def reset(self): self.cache.clear()

    def __repr__(self):
        return 'CacheModel(len={}, size={})'.\
            format(len(self.cache), self.size)
