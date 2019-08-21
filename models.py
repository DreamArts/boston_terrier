import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import pickle
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
import keras
from keras.models import Model
from keras.layers import Input, Dense, Masking, Lambda
from keras.layers.core import Dense, Masking
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform, uniform, orthogonal, TruncatedNormal
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.utils import np_utils, plot_model


# カレントディレクトリのパス
CURRENT_DIR = os.getcwd()
# Word2Vecのサイズ
NUM_VEC = 200
# エンコーダの最大長さ (5, 7, 5)
MAX_ENCODER_LENGTH = 17
# デコーダの最大長さ (5, 7, 5 + 2 * "|" + <EOS>)
MAX_DECODER_LENGTH = 20
# 潜在変数の次元
LATENT_DIM = 256
# バッチサイズ
BATCH_SIZE = 64
# エポック数
EPOCHS = 1
# 学習率
LEARNING_RATE = 0.001


with open('pickle/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('pickle/detokenizer.pickle', 'rb') as g:
    detokenizer = pickle.load(g)
vec_dim = 400
batch_size = 128
input_dim = len(tokenizer) + 2
output_dim = len(tokenizer) + 2
n_hidden = int(vec_dim * 2)  # 隠れ層の次元

# Shimizuのモデル生成クラス
class Dialog_shimizu:
    def __init__(self, maxlen_e=MAX_ENCODER_LENGTH, maxlen_d=MAX_DECODER_LENGTH,
                 n_hidden=LATENT_DIM, input_dim=NUM_VEC, output_dim=NUM_VEC+3):
        self.maxlen_e = maxlen_e  # 入力の単語数 (17)
        self.maxlen_d = maxlen_d  # 出力の単語数 (20)
        self.n_hidden = n_hidden  # 隠れ層の次元
        self.input_dim = input_dim  # エンコーダの単語ベクトルの次元 (200)
        self.output_dim = output_dim  # デコーダの単語ベクトルの次元 (203)

    def create_model(self, encoder_input_data, decoder_input_data, decoder_target_data):
        # エンコーダの定義
        print("Encoder...")
        encoder_inputs = Input(shape=(self.maxlen_e, self.input_dim), dtype="float32", name="encorder_input")
        # Batch Normalization (???)
        e_i = BatchNormalization(axis=-1)(encoder_inputs)
        # Masking (???)
        e_i = Masking(mask_value=0.0)(e_i)
        # 順方向の入力
        e_i_fw0 = e_i
        # 順方向1段目
        e_i_fw1, state_h_fw1, state_c_fw1 = LSTM(self.n_hidden, name="encoder_LSTM_fw1",
                                                 return_sequences=True, return_state=True,
                                                 kernel_initializer=glorot_uniform(seed=20170719),
                                                 recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                )(e_i_fw0)
        # 順方向2段目
        encoder_LSTM_fw2 = LSTM(self.n_hidden, name="encoder_LSTM_fw2",
                                return_sequences=True, return_state=True,
                                kernel_initializer=glorot_uniform(seed=20170719),
                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                dropout=0.5, recurrent_dropout=0.5
                               )
        # 2段目のLSTMに1段目のLSTMの出力を入力
        e_i_fw2, state_h_fw2, state_c_fw2 = encoder_LSTM_fw2(e_i_fw1)
        # 逆方向の入力
        e_i_bw0 = e_i
        # 逆方向1段目
        e_i_bw1, state_h_bw1, state_c_bw1 = LSTM(self.n_hidden, name="encoder_LSTM_bw1",
                                                 return_sequences=True, return_state=True, go_backwards=True,
                                                 kernel_initializer=glorot_uniform(seed=20170719),
                                                 recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                )(e_i_bw0)
        # 逆方向2段目
        e_i_bw2, state_h_bw2, state_c_bw2 = LSTM(self.n_hidden, name="encoder_LSTM_bw2",
                                                 return_sequences=True, return_state=True, go_backwards=True,
                                                 kernel_initializer=glorot_uniform(seed=20170719),
                                                 recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                 dropout=0.5, recurrent_dropout=0.5
                                                )(e_i_bw1)
        # エンコーダの出力
        encoder_outputs = keras.layers.add([e_i_fw2, e_i_bw2], name="encoder_outputs")
        # エンコーダの状態処理
        state_h_1 = keras.layers.add([state_h_fw1, state_h_bw1], name="state_h_1")
        state_c_1 = keras.layers.add([state_c_fw1, state_c_bw1], name="state_c_1")
        state_h_2 = keras.layers.add([state_h_fw2, state_h_bw2], name="state_h_2")
        state_c_2 = keras.layers.add([state_c_fw2, state_c_bw2], name="state_c_2")
        encoder_states1 = [state_h_1, state_c_1]
        encoder_states2 = [state_h_2, state_c_2]
        # エンコーダモデルの処理
        encoder_model = Model(inputs=encoder_inputs,
                              outputs=[encoder_outputs, state_h_1, state_c_1, state_h_2, state_c_2])

        #デコーダ (学習用)
        print("Decoder (learning)...")
        # デコーダを、完全な出力シークエンスを返し、内部状態もまた返すように設定します。
        # 訓練モデルではreturn_sequencesを使用しませんが、推論では使用します。
        a_states1 = encoder_states1
        a_states2 = encoder_states2

        # 1層目のLSTM
        decode_LSTM1 = LSTM(self.n_hidden, name="decode_LSTM1",
                            return_sequences=True, return_state=True,
                            kernel_initializer=glorot_uniform(seed=20170719),
                            recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                           )
        # 2層目のLSTM
        decode_LSTM2 = LSTM(self.n_hidden, name="decode_LSTM2",
                            return_sequences=True, return_state=True,
                            kernel_initializer=glorot_uniform(seed=20170719),
                            recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                            dropout=0.5, recurrent_dropout=0.5
                           )
        # 1層目のDense
        Dense1 = Dense(self.n_hidden,name="Dense1",
                       kernel_initializer=glorot_uniform(seed=20170719))
        # 2層目のDense
        Dense2 = Dense(self.n_hidden, name="Dense2",
                       kernel_initializer=glorot_uniform(seed=20170719))
        # 1回目のConcat (???)
        a_Concat1 = keras.layers.Concatenate(axis=-1)
        # Slice (???)
        a_decode_input_slice1 = Lambda(lambda x: x[:, 0, :], output_shape=(1, self.output_dim,), name="slice1")
        a_decode_input_slice2 = Lambda(lambda x: x[:, 1:, :], name="slice2")
        # Reshape (???)
        a_Reshape1 = keras.layers.Reshape((1, self.output_dim))
        # 1層目のDenseの後のDot
        a_Dot1 = keras.layers.Dot(-1, name="a_Dot1")
        # Softmax層
        a_Softmax=keras.layers.Softmax(axis=-1, name="a_Softmax")
        # 転置
        a_transpose = keras.layers.Reshape((self.maxlen_e, 1), name="Transpose")
        # 転置の後のDot
        a_Dot2 = keras.layers.Dot(1, name="a_Dot2")
        # 2回目のConcat (???)
        a_Concat2 = keras.layers.Concatenate(-1, name="a_Concat2")
        # 3回目のConcat (???) (図にない)
        a_Concat3=keras.layers.Concatenate(axis=-1, name="a_Concat3")
        # デコーダの出力
        decoder_Dense = Dense(self.output_dim, activation="linear", name="decoder_Dense",
                              kernel_initializer=glorot_uniform(seed=20170719))
        # zeros_like (???)
        a_output = Lambda(lambda x: K.zeros_like(x[:, -1, :]), output_shape=(1, self.n_hidden,))(encoder_outputs)
        # Reshape (???)
        a_output = keras.layers.Reshape((1, self.n_hidden))(a_output)

        # デコーダの入力
        decoder_inputs = Input(shape=(self.maxlen_d, self.output_dim), dtype="float32", name="decoder_inputs")
        # Batch Normalization (???)
        d_i = BatchNormalization(axis=-1)(decoder_inputs)
        # Masking (???)
        d_i = Masking(mask_value=0.0)(d_i)
        d_input = d_i

        # よくわかってない部分 (???)
        for i in range(self.maxlen_d):
            d_i_timeslice = a_decode_input_slice1(d_i)
            if i <= self.maxlen_d - 2:
                d_i = a_decode_input_slice2(d_i)
            d_i_timeslice = a_Reshape1(d_i_timeslice)
            lstm_input = a_Concat1([a_output, d_i_timeslice])
            d_i_1, h1, c1 = decode_LSTM1(lstm_input, initial_state=a_states1)
            h_output, h2, c2 = decode_LSTM2(d_i_1, initial_state=a_states2)

            a_states1 = [h1, c1]
            a_states2 = [h2, c2]

            # Attention
            a_o = h_output
            a_o = Dense1(a_o)
            a_o = a_Dot1([a_o, encoder_outputs])  # encoder出力の転置行列を掛ける
            a_o = a_Softmax(a_o)  # Softmax
            a_o = a_transpose(a_o)
            a_o = a_Dot2([a_o, encoder_outputs])  # encoder出力行列を掛ける
            a_o = a_Concat2([a_o, h_output])  # ここまでの計算結果とLSTM出力をconcat
            a_o = Dense2(a_o)
            a_output = a_o  # 次段Attention処理向け出力
            if i == 0 :  # decoder output
                d_output = a_o
            else :
                d_output = a_Concat3([d_output, a_o])
        d_output = keras.layers.Reshape((self.maxlen_d, self.n_hidden))(d_output)

        print("Output...")
        decoder_outputs = decoder_Dense(d_output)
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        opt = optimizers.RMSprop(lr=LEARNING_RATE)
        model.compile(loss="mean_squared_error",optimizer=opt)
        es_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=0, mode="auto")
        if self.maxlen_d != 1:
            model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05, callbacks=[es_cb])

        #デコーダ (川柳生成)
        print("Decoder (senryu)...")
        decoder_state_input_h_1 = Input(shape=(self.n_hidden,), name="input_h_1")
        decoder_state_input_c_1 = Input(shape=(self.n_hidden,), name="input_c_1")
        decoder_state_input_h_2 = Input(shape=(self.n_hidden,), name="input_h_2")
        decoder_state_input_c_2 = Input(shape=(self.n_hidden,), name="input_c_2")
        decoder_states_inputs_1 = [decoder_state_input_h_1, decoder_state_input_c_1]
        decoder_states_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]
        decoder_states_inputs = [decoder_state_input_h_1, decoder_state_input_c_1,
                                 decoder_state_input_h_2, decoder_state_input_c_2]
        decoder_input_c = Input(shape=(1, self.n_hidden), name="decoder_input_c")
        decoder_input_encoded = Input(shape=(self.maxlen_e, self.n_hidden), name="decoder_input_encoded")
        # LSTM 1段目
        decoder_i_timeslice = a_Reshape1(a_decode_input_slice1(d_input))
        l_input = a_Concat1([decoder_input_c, decoder_i_timeslice])  # 前段出力とdcode_inputをconcat
        decoder_lstm_1,state_h_1, state_c_1 = decode_LSTM1(l_input,
                                                           initial_state=decoder_states_inputs_1)  # initial_stateが学習の時と違う
        # LSTM 2段目
        decoder_lstm_2, state_h_2, state_c_2 = decode_LSTM2(decoder_lstm_1,
                                                            initial_state=decoder_states_inputs_2)
        decoder_states = [state_h_1, state_c_1, state_h_2, state_c_2]

        # Attention
        attention_o = Dense1(decoder_lstm_2)
        attention_o = a_Dot1([attention_o, decoder_input_encoded])  # encoder出力の転置行列を掛ける
        attention_o = a_Softmax(attention_o)  # Softmax
        attention_o = a_transpose (attention_o)
        attention_o = a_Dot2([attention_o, decoder_input_encoded])  # encoder出力行列を掛ける
        attention_o = a_Concat2([attention_o, decoder_lstm_2])  # ここまでの計算結果とLSTM出力をconcat

        attention_o = Dense2(attention_o)
        decoder_o = attention_o

        print("Output...")
        decoder_res = decoder_Dense(decoder_o)
        decoder_model = Model([decoder_inputs, decoder_input_c, decoder_input_encoded] + decoder_states_inputs,
                              [decoder_res, decoder_o] + decoder_states)
        return model, encoder_model, decoder_model




# Watanabeのモデル生成クラス
class Dialog_watanabe:
    def __init__(self, maxlen_e=MAX_ENCODER_LENGTH, maxlen_d=1, n_hidden=n_hidden,input_dim=input_dim,vec_dim=vec_dim,output_dim=output_dim,tokenizer=tokenizer,detokenizer=detokenizer):
        self.maxlen_e = maxlen_e
        self.maxlen_d = maxlen_d
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.vec_dim = vec_dim
        self.output_dim = output_dim
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer

    def create_model(self):
        #エンコーダー
        encoder_input = Input(shape=(self.maxlen_e,), dtype='int32', name='encorder_input')
        
        e_i = Embedding(output_dim=self.vec_dim, input_dim=self.input_dim, #input_length=self.maxlen_e,
                        mask_zero=True, 
                        embeddings_initializer=uniform(seed=20170719))(encoder_input)
        

        e_i = BatchNormalization(axis=-1)(e_i)
        e_i = Masking(mask_value=0.0)(e_i)

        e_i_fw1, state_h_fw1, state_c_fw1 = LSTM(self.n_hidden, name='encoder_LSTM_fw1'  , #前向き1段目
                                                return_sequences=True,return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                #dropout=0.5, recurrent_dropout=0.5
                                                )(e_i) 
        encoder_LSTM_fw2 = LSTM(self.n_hidden, name='encoder_LSTM_fw2'  ,       #前向き2段目
                                                return_sequences=True,return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                dropout=0.5, recurrent_dropout=0.5
                                                )  

        e_i_fw2, state_h_fw2, state_c_fw2 = encoder_LSTM_fw2(e_i_fw1)
        e_i_bw0 = e_i
        e_i_bw1, state_h_bw1, state_c_bw1 = LSTM(self.n_hidden, name='encoder_LSTM_bw1'  ,  #後ろ向き1段目
                                                return_sequences=True,return_state=True, go_backwards=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                #dropout=0.5, recurrent_dropout=0.5
                                                )(e_i_bw0) 
        e_i_bw2, state_h_bw2, state_c_bw2 = LSTM(self.n_hidden, name='encoder_LSTM_bw2'  ,  #後ろ向き2段目
                                                return_sequences=True,return_state=True, go_backwards=True,
                                                kernel_initializer=glorot_uniform(seed=20170719), 
                                                recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                                                dropout=0.5, recurrent_dropout=0.5
                                                )(e_i_bw1)            

        encoder_outputs = keras.layers.add([e_i_fw2,e_i_bw2],name='encoder_outputs')
        state_h_1 = keras.layers.add([state_h_fw1,state_h_bw1],name='state_h_1')
        state_c_1 = keras.layers.add([state_c_fw1,state_c_bw1],name='state_c_1')
        state_h_2 = keras.layers.add([state_h_fw2,state_h_bw2],name='state_h_2')
        state_c_2 = keras.layers.add([state_c_fw2,state_c_bw2],name='state_c_2')
        encoder_states1 = [state_h_1,state_c_1] 
        encoder_states2 = [state_h_2,state_c_2]

        encoder_model = Model(inputs=encoder_input, 
                                outputs=[encoder_outputs,state_h_1,state_c_1,state_h_2,state_c_2])        

   
        # デコーダー（学習用）
        # デコーダを、完全な出力シークエンスを返し、内部状態もまた返すように設定します。
        # 訓練モデルではreturn_sequencesを使用しませんが、推論では使用します。
        a_states1 = encoder_states1
        a_states2 = encoder_states2

        #レイヤー定義
        decode_LSTM1 = LSTM(self.n_hidden, name='decode_LSTM1',
                            return_sequences=True, return_state=True,
                            kernel_initializer=glorot_uniform(seed=20170719), 
                            recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                            )
        decode_LSTM2 = LSTM(self.n_hidden, name='decode_LSTM2',
                            return_sequences=True, return_state=True,
                            kernel_initializer=glorot_uniform(seed=20170719), 
                            recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                            dropout=0.5, recurrent_dropout=0.5
                            )                  

        Dense1 = Dense(self.n_hidden,name='Dense1',
                            kernel_initializer=glorot_uniform(seed=20170719))
        Dense2 = Dense(self.n_hidden,name='Dense2',     #次元を減らす
                            kernel_initializer=glorot_uniform(seed=20170719))              
        a_Concat1 = keras.layers.Concatenate(axis=-1)
        a_decode_input_slice1 = Lambda(lambda x: x[:,0,:],output_shape=(1,self.vec_dim,),name='slice1')
        a_decode_input_slice2 = Lambda(lambda x: x[:,1:,:],name='slice2')
        a_Reshape1 = keras.layers.Reshape((1,self.vec_dim))
        a_Dot1 = keras.layers.Dot(-1,name='a_Dot1')
        a_Softmax = keras.layers.Softmax(axis=-1,name='a_Softmax')
        a_transpose = keras.layers.Reshape((self.maxlen_e,1),name='Transpose')
        a_Dot2 = keras.layers.Dot(1,name='a_Dot2')
        a_Concat2 = keras.layers.Concatenate(-1,name='a_Concat2')
        a_tanh = Lambda(lambda x: K.tanh(x),name='tanh')
        a_Concat3 = keras.layers.Concatenate(axis=-1,name='a_Concat3')
        decoder_Dense = Dense(self.output_dim,activation='softmax', name='decoder_Dense',
                                kernel_initializer=glorot_uniform(seed=20170719))        

        a_output = Lambda(lambda x: K.zeros_like(x[:,-1,:]),output_shape=(1,self.n_hidden,))(encoder_outputs) 
        a_output = keras.layers.Reshape((1,self.n_hidden))(a_output)

        decoder_inputs = Input(shape=(self.maxlen_d,), dtype='int32', name='decorder_inputs')        
        d_i = Embedding(output_dim=self.vec_dim, input_dim=self.input_dim, #input_length=self.maxlen_d,
                        mask_zero=True,
                        embeddings_initializer=uniform(seed=20170719))(decoder_inputs)
        d_i = BatchNormalization(axis=-1)(d_i)
        d_i = Masking(mask_value=0.0)(d_i)    
        # d_i = Lambda(lambda x: 0.01*x)(d_i)       
        d_input = d_i

        for i in range(self.maxlen_d):
            d_i_timeslice = a_decode_input_slice1(d_i)
            if i <= self.maxlen_d-2:
                d_i = a_decode_input_slice2(d_i)
            d_i_timeslice = a_Reshape1(d_i_timeslice)
            lstm_input = a_Concat1([a_output,d_i_timeslice])  # 前段出力とdcode_inputをconcat
            d_i_1, h1, c1 = decode_LSTM1(lstm_input,initial_state=a_states1) 
            h_output, h2, c2 = decode_LSTM2(d_i_1,initial_state=a_states2)            

            a_states1 = [h1,c1]
            a_states2 = [h2,c2]

            #attention
            a_o = h_output
            a_o = Dense1(a_o)
            a_o = a_Dot1([a_o,encoder_outputs])  # encoder出力の転置行列を掛ける
            a_o = a_Softmax(a_o)  # softmax
            a_o = a_transpose (a_o) 
            a_o = a_Dot2([a_o,encoder_outputs])  # encoder出力行列を掛ける
            a_o = a_Concat2([a_o,h_output])  # ここまでの計算結果とLSTM出力をconcat
            a_o = Dense2(a_o)  
            a_o = a_tanh(a_o)  # tanh
            a_output = a_o  # 次段attention処理向け出力
            if i == 0:  # docoder_output
                d_output = a_o
            else:
                d_output = a_Concat3([d_output,a_o]) 

        d_output = keras.layers.Reshape((self.maxlen_d, self.n_hidden))(d_output)        
        decoder_outputs = decoder_Dense(d_output)
        model = Model(inputs=[encoder_input, decoder_inputs], outputs=decoder_outputs)
        model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['categorical_accuracy'])

        #デコーダー（応答文作成）
        decoder_state_input_h_1 = Input(shape=(self.n_hidden,),name='input_h_1')
        decoder_state_input_c_1 = Input(shape=(self.n_hidden,),name='input_c_1')
        decoder_state_input_h_2 = Input(shape=(self.n_hidden,),name='input_h_2')
        decoder_state_input_c_2 = Input(shape=(self.n_hidden,),name='input_c_2')        
        decoder_states_inputs_1 = [decoder_state_input_h_1, decoder_state_input_c_1]
        decoder_states_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]  
        decoder_states_inputs = [decoder_state_input_h_1, decoder_state_input_c_1,
                                decoder_state_input_h_2, decoder_state_input_c_2]
        decoder_input_c = Input(shape=(1,self.n_hidden),name='decoder_input_c')
        decoder_input_encoded = Input(shape=(self.maxlen_e,self.n_hidden),name='decoder_input_encoded')
        #LSTM１段目
        decoder_i_timeslice = a_Reshape1(a_decode_input_slice1(d_input))
        l_input = a_Concat1([decoder_input_c, decoder_i_timeslice])      #前段出力とdcode_inputをconcat
        decoder_lstm_1,state_h_1, state_c_1  = decode_LSTM1(l_input,
                                                        initial_state=decoder_states_inputs_1)  #initial_stateが学習の時と違う
        #LSTM２段目
        decoder_lstm_2, state_h_2, state_c_2  = decode_LSTM2(decoder_lstm_1,
                                                        initial_state=decoder_states_inputs_2) 
        decoder_states = [state_h_1,state_c_1,state_h_2, state_c_2]

        #attention
        attention_o = Dense1(decoder_lstm_2)
        attention_o = a_Dot1([attention_o, decoder_input_encoded])                   #encoder出力の転置行列を掛ける
        attention_o = a_Softmax(attention_o)                                         #softmax
        attention_o = a_transpose (attention_o) 
        attention_o = a_Dot2([attention_o, decoder_input_encoded])                    #encoder出力行列を掛ける
        attention_o = a_Concat2([attention_o, decoder_lstm_2])                        #ここまでの計算結果とLSTM出力をconcat
        attention_o = Dense2(attention_o)  
        decoder_o = a_tanh(attention_o)                                               #tanh
        decoder_res = decoder_Dense(decoder_o)
        decoder_model = Model(
        [decoder_inputs,decoder_input_c,decoder_input_encoded] + decoder_states_inputs,
        [decoder_res, decoder_o] + decoder_states)                                           

        return model, encoder_model, decoder_model