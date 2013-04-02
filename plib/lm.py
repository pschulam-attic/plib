"""
Utilities for working with language models.
"""

import math
import sys
from itertools import chain as _chain

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

    nmodels = len(likelihood_lists)
    assert nmodels > 1
    nwords = len(likelihood_lists[0])
    assert all(nwords == len(l) for l in likelihood_lists)
    weights = [1. / nmodels] * nmodels
    converged = False

    def log_likelihood(weights, likelihood_lists):
        ll = 0.0
        for probs in zip(*likelihood_lists):
            ll += math.log(sum(w * p for w, p in zip(weights, probs)), 10)
        return ll

    while not converged:
        old_ll = log_likelihood(weights, likelihood_lists)
        new_weights = list(weights)
        partitions = [sum(w * p for w, p in zip(weights,probs))
                      for probs in zip(*likelihood_lists)]
        for i, likelihoods in enumerate(likelihood_lists):
            posterior = 0.0
            for j, p in enumerate(likelihoods):
                posterior += (weights[i] * p) / partitions[j]
            new_weights[i] = posterior / nwords

        weights = new_weights
        new_ll = log_likelihood(weights, likelihood_lists)
        if verbose:
            sys.stderr.write('interpolate: LL= {:.4f} ( delta= {:.4f} )\n'.format(new_ll, abs(new_ll-old_ll)))
        if abs(new_ll - old_ll) < tol:
            converged = True

    return weights
