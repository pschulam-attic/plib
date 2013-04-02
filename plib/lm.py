"""
Utilities for working with language models.
"""

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
