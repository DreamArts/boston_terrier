# -*- coding:utf-8 -*-
"""
summary.py
要約を行う

LexRankを利用
"""
import json
from sudachipy import dictionary
from sudachipy import tokenizer
import numpy as np

from gensim.models.doc2vec import Doc2Vec
from embededrank import EmbedRank
from nlp_util import tokenize

# tokenizer_object init
tokenizer_obj = dictionary.Dictionary(config_path="conf/sudachi.json").create()
#summarizer = TextRankSummarizer2()
#summarizer.stop_words = [" "]
model = Doc2Vec.load("models/jawiki.doc2vec.dbow300d.model")
embedrank = EmbedRank(model=model, tokenize=tokenize)


def summarize(sentence, sentences_count=1, word_count=17,  method="EmbedRank"):
    """
    要約をする

    ARGS:
      sentence  要約を行う文

    RETURNS:
      summary  要約文
    """
    if method == "TextRank":
        from tokenizers import Tokenizer
        from text_rank_w2v import TextRankSummarizer2
        from sumy.parsers.plaintext import PlaintextParser
        from text_rank_w2v import TextRankSummarizer2

        summarizer = TextRankSummarizer2()
        summarizer.stop_words = [" "]

        parser = PlaintextParser.from_string(sentence, Tokenizer("sudachi"))
        try:
            summary = summarizer(document=parser.document,
                                 sentences_count=sentences_count)
        except:
            summary = []
        if len(summary) == 0:
            return _tokenize(sentence, "words")[:word_count]
        return _tokenize(str(summary), "words")[:word_count]
    elif method == "EmbedRank":
        #model = Doc2Vec.load("models/jawiki.doc2vec.dbow300d.model")
        #embedrank = EmbedRank(model=model, tokenize=tokenize)
        ret = embedrank.extract_keyword(sentence)
        ret = [a for a, b in ret][:word_count]
        if len(ret) == 0:
            return _tokenize(sentence, "words")[:word_count]
        return ret
    else:
        return []


def interactive_summarize(sentences):
    """

    """
    return "対話形式の要約文を返します．"


def _tokenize(sentence, ret="surface", parts_of_speechs=["名詞", "動詞", "感動詞"], mode=tokenizer.Tokenizer.SplitMode.C):
    tokens = tokenizer_obj.tokenize(
        sentence, mode)
    # 名詞と同士のリスト
    if ret == "surface":
        return [l.surface() for l in tokens]
    elif ret == "words":
        return [l.surface() for l in tokens if l.part_of_speech()[0] in parts_of_speechs]
    elif ret == "tokens":
        return tokens
    else:
        return []


if __name__ == "__main__":
    """
    テスト関数
    """
    import sys
    sentence = sys.argv[1]
    s1 = summarize(sentence, method="EmbedRank")
    print(s1)
    #s2 = summarize(sentence, method="TextRank")
    # print(s2)
    # for sentence in summary:
    #    print(sentence)
