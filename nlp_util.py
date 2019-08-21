import sudachipy
from sudachipy import dictionary
from sudachipy import tokenizer

tokenizer_obj = dictionary.Dictionary(config_path="conf/sudachi.json").create()


def tokenize(text, ret="surface", parts_of_speechs=["名詞", "動詞", "形容詞"]):
    tokens = tokenizer_obj.tokenize(
        text, tokenizer.Tokenizer.SplitMode.C)
    # 名詞と動詞のリスト
    if ret == "surface":
        return [l.surface() for l in tokens if l.part_of_speech()[0] in parts_of_speechs]
    elif ret == "words":
        return [l.normalized_form() for l in tokens if l.part_of_speech()[0] in parts_of_speechs]
    elif ret == "tokens":
        return tokens
    else:
        return []


def extract_keyphrase_candidates(text):
    tokens = tokenizer_obj.tokenize(
        text, tokenizer.Tokenizer.SplitMode.A)

    keyphrase_candidates = []
    phrase = []
    phrase_noun = []
    is_adj_candidate = False
    is_multinoun_candidate = False
    index = 0

    while index != len(tokens):
        pom = tokens[index].part_of_speech()[0]
        surface = tokens[index].surface()
        if pom == "形容詞":
            is_adj_candidate = True
            phrase.append(surface)
        if pom == "名詞" and is_adj_candidate:
            phrase.append(surface)
        elif len(phrase) >= 2:
            keyphrase_candidates.append(phrase)

        is_adj_candidate = False
        phrase = []

        if pom.startswith("名詞"):
            phrase_noun.append(surface)
            is_multinoun_candidate = True
        elif len(phrase_noun) >= 2:
            keyphrase_candidates.append(phrase_noun)
            is_multinoun_candidate = False
            phrase_noun = []
        else:
            is_multinoun_candidate = False
            phrase_noun = []

        index += 1

    return keyphrase_candidates
