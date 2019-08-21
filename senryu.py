# -*- coding:utf-8 -*-

"""
キーワードからJapanese Traditional Senryuを生成したい！
"""


import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pickle
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
import MeCab
from keras.models import Model, load_model
import models


# カレントディレクトリのパス
CURRENT_DIR = os.getcwd()
# word2vecの学習済モデルパス
WORD2VEC_PATH = os.path.join(CURRENT_DIR, "entity_vector/entity_vector.model.bin")
# tokenizerのパス
TOKENIZER_PATH = os.path.join(CURRENT_DIR, "pickle/tokenizer.pickle")
# detokenizerのパス
DETOKENIZER_PATH = os.path.join(CURRENT_DIR, "pickle/detokenizer.pickle")
# 学習済重みパス (Shimizu)
WEIGHT_PATH_S = os.path.join(CURRENT_DIR, "model/weight_shimizu.h5")
# 学習済重みパス (Watanabe)
WEIGHT_PATH_W = os.path.join(CURRENT_DIR, "model/weight_watanabe.h5")
# MeCabのTagger設定
TAGGER = MeCab.Tagger()
# Word2Vecのサイズ
NUM_VEC = 200
# エンコーダの最大長さ (5, 7, 5)
MAX_ENCODER_LENGTH = 17
# デコーダの最大長さ (5, 7, 5 + 2 * "|" + <EOS>)
MAX_DECODER_LENGTH = 20
# 潜在変数の次元
LATENT_DIM = 256
# 拍数のリスト
COBS = (5, 7, 5)
# 小文字のリスト
SMALL_LETTERS = ["ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ"]


def senryu(words: list, mode: str="S", encoder_model: Model=None, decoder_model: Model=None, is_load_weight: bool=False, word2vec_model: Word2VecKeyedVectors=None, tokenizer: dict=None, detokenizer: dict=None) -> str:
    """
    単語のリストから川柳を生成
    @param words: キーワードのリスト
    @param mode: 予測モード ("S": Shimizu作, "W": Watanabe作)
    @param encoder_model: エンコーダモデル
    @param decoder_model: デコーダモデル
    @param is_load_weight: モデルが学習済か判定するフラグ
    @param word2vec_model: Word2Vecモデル
    @param tokenizer: word -> id
    @param detokenizer: id -> word
    @return senryu: 川柳
    """
    senryu = None
    # Shimizu
    if mode == "S":
        if word2vec_model is None:
            word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        if encoder_model is None or decoder_model is None:
            dialog = models.Dialog_shimizu(maxlen_d=1)
            model, encoder_model, decoder_model = dialog.create_model(None, None, None)
        if not is_load_weight:
            model.load_weights(WEIGHT_PATH_S)
        input_seq = generate_sequence(words, word2vec_model)
        senryu = generate_senryu(input_seq, encoder_model, decoder_model, word2vec_model)
    # Watanabe
    elif mode == "W":
        if tokenizer is None:
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
        if detokenizer is None:
            with open(DETOKENIZER_PATH, 'rb') as g:
                detokenizer = pickle.load(g)
        if encoder_model is None or decoder_model is None:
            dialog = models.Dialog_watanabe()
            model, encoder_model, decoder_model = dialog.create_model()
        if not is_load_weight:
            model.load_weights(WEIGHT_PATH_W)
        senryu = predict_senryu(words, encoder_model, decoder_model, tokenizer, detokenizer)
    # Others
    else:
        print("MODE: {} is NOT defined.".format(mode))
    print("入力した単語リスト: {}".format(words))
    print("生成した川柳: {}".format(senryu))
    return senryu


def generate_sequence(words: list, word2vec_model: Word2VecKeyedVectors) -> np.ndarray:
    """
    単語列からシーケンスを生成
    @param words: 単語リスト
    @return sequence: シーケンス
    """
    # シーケンスの初期化
    sequence = np.zeros((1, MAX_ENCODER_LENGTH, NUM_VEC))
    # 単語をWord2Vecでベクトルに変換
    for i, word in enumerate(words):
        try:
            vec = word2vec_model[word]
            sequence[0][-i-1] = vec
        # その単語がWord2Vecモデルに存在しない場合、終了
        except KeyError:
            print("{} is NOT found.".format(word))
            # sys.exit()
    return sequence


def generate_senryu(input_seq: np.ndarray, encoder_model: Model, decoder_model: Model, word2vec_model: Word2VecKeyedVectors) -> str:
    """
    学習したSeq2Seqモデルで、入力された単語列から川柳を生成
    @param input_seq: 入力する単語列をベクトル表記したテンソル
    @param encoder_model: 学習済エンコーダモデル
    @param decoder_model: 学習済デコーダモデル
    @param word2vec_model: Word2Vecモデル
    @return senryu: 生成された川柳
    """
    # エンコーダのinputである単語列を入力し、エンコーダモデルの出力を取得
    encoder_outputs, state_h1, state_c1, state_h2, state_c2 = encoder_model.predict(input_seq)
    states_value = [state_h1, state_c1, state_h2, state_c2]
    # (???)
    decoder_input_c = encoder_outputs[:, -1, :].reshape((1, 1, LATENT_DIM))
    # デコーダの最初のinputである<EOS>を宣言
    target_seq = np.zeros((1, 1, NUM_VEC+3))
    # 逐次単語を生成し、川柳を生成
    # 出力を終了するフラグ
    stop_condition = False
    # 生成された川柳
    senryu = ""
    # 1つ前に予測された単語の品詞
    pos_before = ""
    # 1つ前に予測された単語
    pred_word_before = ""
    # トータルの拍数
    cob_total = 0
    # 現在の句の拍数
    cob = 0
    # 現在の句のインデクス
    index_cob = 0
    while not stop_condition:
        # デコーダモデルで単語を生成
        output_tokens, d_output, h1, c1, h2, c2 = decoder_model.predict([target_seq, decoder_input_c, encoder_outputs] + states_value)
        # 状態更新
        decoder_input_c = d_output
        states_value = [h1, c1, h2, c2]
        # ベクトル部を取得
        output_tokens = output_tokens[0][0][:NUM_VEC]
        # 予測したベクトルに最も類似した単語を取得
        pred_word = word2vec_model.most_similar([output_tokens], [], 1)[0][0]
        # 予測された単語が"|"の場合、形態素解析は行わない
        if pred_word == "|":
            if pred_word_before != "|":
                # 現在の拍数をリセット
                cob = 0
                # 拍数のインデクスを進める
                index_cob += 1
        # 予測された単語が"|"以外の場合、形態素解析してフリガナの文字数をカウント
        else:
            # 予測した単語を形態素分析
            keywords = TAGGER.parse(pred_word)
            # すべての単語について走査 (予測した単語と<EOS>の2つであることが望ましい)
            for keyword in keywords.split("\n"):
                # 単語
                word = keyword.split("\t")[0]
                # End of Sequenceの場合、ループを抜ける
                if word == "EOS":
                    break
                # 単語に関する情報
                infos = keyword.split("\t")[1].split(",")
                # 品詞
                pos = infos[0]
                # フリガナ
                try:
                    furigana = infos[7]
                except IndexError:
                    print("フリガナ is NOT found. [{}]".format(len(infos)))
                    print("Word: {} | Infos: {}".format(word, infos))
                    continue
                if pos in ["助詞", "接続詞"] and pos_before in ["助詞", "接続詞"]:
                    pass
                else:
                    # フリガナの文字数だけ拍数を加算
                    cob_total += len([letter for letter in furigana if not letter in SMALL_LETTERS])
                    cob += len([letter for letter in furigana if not letter in SMALL_LETTERS])
                    cob = min(cob, *COBS)
        # 現在の拍数を2進数に変換
        cob_bin = bin(cob)[2:]
        # 現在の拍数が指定拍数を超えた場合
        if index_cob < 3:
            if cob > COBS[index_cob]:
                # 現在の拍数をリセット
                cob = 0
                # 拍数のインデクスを進める
                index_cob += 1
        # 全拍数が17を超えた場合、予測を終了
        if cob_total >= MAX_ENCODER_LENGTH or index_cob >= 3:
            stop_condition = True
        # 末尾に追加する拍数情報
        append = [0, 0, 0]
        for i, b in enumerate(cob_bin[::-1]):
            append[-i-1] = int(b)
        try:
            # 次の入力シーケンスを生成
            target_seq[0][0] = np.array(word2vec_model[word].tolist() + append)
        except KeyError:
            print("Fail to generate 川柳")
            sys.exit()
        if pred_word == "|" and pred_word_before == "|":
            pass
        elif pos in ["助詞", "接続詞"] and pos_before in ["助詞", "接続詞"] and pred_word != "|":
            pass
        else:
            # 生成する川柳に予測した単語を追加
            senryu += pred_word
        # 予測した単語を保存
        pred_word_before = pred_word
        pos_before = pos
    senryu = senryu.replace("|", " ")
    return senryu


def predict_senryu(words: list, encoder_model: Model, decoder_model: Model, tokenizer: dict, detokenizer: dict) -> str:
    '''
    入力された単語列から川柳を生成
    @param words: キーワードのリスト
    @param encoder_model: 学習済エンコーダモデル
    @param decoder_model: 学習済デコーダモデル
    @tokenizer: 単語 -> ID
    @detokenizer: ID -> 単語
    @return senryu: 生成された川柳
    '''
    VEC_DIM = 400
    INPUT_DIM = len(tokenizer)
    OUTPUT_DIM = len(tokenizer)
    ENC_MAXLEN = 17
    DEC_MAXLEN = 20
    N_HIDDEN = int(VEC_DIM*2)


    # 入力の単語列に学習コーパスに存在しない単語があった場合、除く
    corpus = list(tokenizer.keys())
    for i, word in enumerate(words[:]):
        if word not in corpus:
            print(word, 'is not in the corpus.', word, 'is replaced to zero-padding.')
            words[i] = '\t'

    # 入力の単語列をIDに変換
    # print(words)
    words = [tokenizer[word] for word in words]
    corpus_words = list(tokenizer.keys())
    enc_input = np.zeros((1, ENC_MAXLEN))
    for i, word in enumerate(words):
        if word in corpus_words:
            enc_input[0, i] = tokenizer[word]
        else:
            enc_input[0, i] = 0  # 未知の単語の場合、0(padding)とする

    encoder_outputs, state_h_1, state_c_1, state_h_2, state_c_2 = encoder_model.predict(enc_input)
    states_value = [state_h_1, state_c_1, state_h_2, state_c_2]

    # Generate empty target sequence of length 1. -> Modified to have length 20.
    target_seq = np.zeros((1, 1))
    # print(target_seq.shape)
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokenizer['<EOS>']
    decoder_input_c = encoder_outputs[:,-1,:].reshape((1, 1, N_HIDDEN))

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # stop_condition = False
    senryu = []
    # print(words)
    for i in range(DEC_MAXLEN):
        # print(target_seq.shape)
        output_tokens, d_output, h1, c1,h2,c2 = decoder_model.predict(
            [target_seq, decoder_input_c, encoder_outputs] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_char = detokenizer[sampled_token_index]

        # Exit condition: find stop character.
        if sampled_char == '<EOS>':
            break
        senryu.append(sampled_char)
        # Update the target sequence (of length 1).
        if i == DEC_MAXLEN:
            break
        target_seq[0, 0] = sampled_token_index
        decoder_input_c = d_output
        # Update states
        states_value = [h1, c1, h2, c2]

    # print('DECODER OUTPUT: ', senryu)
    senryu = ''.join(senryu)
    senryu = senryu.replace('|', ' ')
    # print('RETURN SENRYU', senryu)
    return senryu


if __name__ == "__main__":
    senryu(sys.argv[1:], mode="S")
