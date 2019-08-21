# -*- coding:utf-8 -*-

from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import math
import numpy
from gensim.models import word2vec

w2v_model = word2vec.Word2Vec.load("models/update.bin")


class TextRankSummarizer2(TextRankSummarizer):

    @staticmethod
    def _rate_sentences_edge(words1, words2):
        rank = 0
        for w1 in words1:
            for w2 in words2:
                if w1 in w2v_model.wv and w2 in w2v_model.wv:
                    rank += w2v_model.similarity(w1, w2)
                else:
                    rank += int(w1 == w2)
        if rank == 0:
            return 0.0

        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        if numpy.isclose(norm, 0.):
                # This should only happen when words1 and words2 only have a single word.
                # Thus, rank can only be 0 or 1.
            assert rank in (0, 1)
            return rank * 1.0
        else:
            return rank / norm


def add_vocab(inputText):
    sentences = word2vec.Text8Corpus(inputText)
    w2v_model.build_vocab(sentences, update=True)
    w2v_model.train(
        sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
    w2v_model.save("models/update.bin")


if __name__ == "__main__":
    add_vocab("corps.txt")
