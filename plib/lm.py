"""
Utilities for working with language models.
"""

from itertools import chain as _chain

def parse_srilm_ppl(stream):
    """
    Parse a perplexity file produced by the SRILM toolkit at debugging
    level 2. Log probabilities are given in base 10.

    Returns a 2-tuple, with entries:

      (1) List of sentence_scores, where a sentence score is a list of
      tuples that can be destructured using (word, lmword, ll). Word
      is the word from the original sentence, lmword is the word that
      was seen by the LM (could be <unk> for example), and ll is the
      log-likelihood (base 10).

      (2) A dictionary containing summary information from the ppl
      file. This contains, for example, the total number of sentences,
      the total number of tokens, the perplexity, and the full
      log-likelihood.

    """
    def parse_word_score(s, WORD=1, LOGP=-2):
        fields = s.strip().split()
        return fields[WORD], float(fields[LOGP])

    def parse_sentence_score(s):
        lines = s.strip().split('\n')
        sentence = lines[0].split() + ['</s>']
        word_probs = (parse_word_score(l) for l in lines if l.startswith('\t'))
        return [(w,wlm,ll) for w, (wlm, ll) in zip(sentence, word_probs)]

    def parse_summary(s, NSENTS=2, NWORDS=4, LPROB=3, PPL=5):
        corpus_summary, model_summary = s.strip().split('\n')
        nsents = corpus_summary.split()[NSENTS]
        nwords = corpus_summary.split()[NWORDS]
        ll = float(model_summary.split()[LPROB])
        ppl = float(model_summary.split()[PPL])
        return dict(nsents=nsents, nwords=nwords, ll=ll, ppl=ppl)

    chunks = stream.read().strip().split('\n\n')
    sentence_scores = [parse_sentence_score(s) for s in chunks[:-1]]
    summary = parse_summary(chunks[-1])
    return sentence_scores, summary
