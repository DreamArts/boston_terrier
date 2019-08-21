# -*- coding:utf-8 -*-

import os
import json
from datetime import datetime
import pickle
from flask import Flask, request
from dotenv import load_dotenv
import requests
from gensim.models import KeyedVectors

# load other files
import models
from summary import summarize
from senryu import senryu
from preprocess import normalize_neologd

app = Flask(__name__)
load_dotenv('.env')
env = os.environ
TRIGGER_MSG_S = '575s'
TRIGGER_MSG_W = '575w'
GET_MSG_DIF = 60*60  # 60秒前から、過去のメッセージを取得
GET_MAX_MSG = 5  # 最大何件分のメッセージをとってくるか指定
MODE = 'S'
MODE_SUMMARY = 'EmbedRank'

CURRENT_DIR = os.getcwd()
WORD2VEC_PATH = os.path.join(CURRENT_DIR, 'entity_vector/entity_vector.model.bin')
TOKENIZER_PATH = os.path.join(CURRENT_DIR, 'pickle/tokenizer.pickle')
DETOKENIZER_PATH = os.path.join(CURRENT_DIR, 'pickle/detokenizer.pickle')
WEIGHT_SHIMIZU_PATH = os.path.join(CURRENT_DIR, 'model/weight_shimizu.h5')
WEIGHT_WATANABE_PATH = os.path.join(CURRENT_DIR, 'model/weight_watanabe.h5')

dialog_shimizu = models.Dialog_shimizu(maxlen_d=1)
model_shimizu, encoder_model_shimizu, decoder_model_shimizu = dialog_shimizu.create_model(None, None, None)
model_shimizu.load_weights(WEIGHT_SHIMIZU_PATH)
word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)

dialog_watanabe = models.Dialog_watanabe()
model_watanabe, encoder_model_watanabe, decoder_model_watanabe = dialog_watanabe.create_model()
# model_watanabe.load_weights(WEIGHT_WATANABE_PATH, by_name=True)
model_watanabe.load_weights(WEIGHT_WATANABE_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(DETOKENIZER_PATH, 'rb') as g:
    detokenizer = pickle.load(g)

print('----LOADED----')
print()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@ app.route('/message', methods=['POST'])
def messages():
    if is_request_valid(request):
        body = request.get_json(silent=True)
        companyId = body['companyId']
        msgObj = body['message']
        groupId = msgObj['groupId']
        messageText = msgObj['text']
        userName = msgObj['createdUserName']
        
        print('GET MESSAGE', messageText)
        if is_message_trigger_s(messageText):
            print('---MESSAGE IS A TRIGGER---')
            target_text = get_target_text(companyId, groupId)
            print('TARGET TEXT:', target_text)
            normalized_target_text = normalize_neologd(target_text)
            print('NORMALIZED TARGET TEXT:', normalized_target_text)
            haiku = main(mode='S', target_text=normalized_target_text)
            # haiku = normalized_target_text
            send_message(companyId, groupId, haiku)
        elif is_message_trigger_w(messageText):
            print('---MESSAGE IS A TRIGGER---')
            target_text = get_target_text(companyId, groupId)
            print('TARGET TEXT:', target_text)
            normalized_target_text = normalize_neologd(target_text)
            print('NORMALIZED TARGET TEXT:', normalized_target_text)
            haiku = main(mode='W', target_text=normalized_target_text)
            # haiku = normalized_target_text
            send_message(companyId, groupId, haiku)
        return "OK"
        
    else:
        return "Request is not valid."

# Check if token is valid.
def is_request_valid(request):
    validationToken = env['CHIWAWA_VALIDATION_TOKEN']
    requestToken = request.headers['X-Chiwawa-Webhook-Token']
    return validationToken == requestToken

# Check if a message is a trigger word.
def is_message_trigger_s(message):
    return message == TRIGGER_MSG_S

def is_message_trigger_w(message):
    return message == TRIGGER_MSG_W

# Get a target text via Chiwawa WebAPI
def get_target_text(companyId, groupId):
    # createdAtFrom = int(datetime.now().timestamp() - GET_MSG_DIF)
    url = 'https://{0}.chiwawa.one/api/public/v1/groups/{1}/messages'.format(companyId, groupId)
    headers = {
        'Content-Type': 'application/json',
        'X-Chiwawa-API-Token': env['CHIWAWA_API_TOKEN']
    }
    params = {
        # 'createdAtFrom': createdAtFrom, 
        # 'createdAtTo': {どの時刻までのメッセージを取得するか指定}
        'maxResults': GET_MAX_MSG
    }
    r = requests.get(url=url, headers=headers, params=params)
    r = r.json()
    target_text = r['messages'][1]['text']
    return target_text

# Send message to Chiwawa server
def send_message(companyId, groupId, message):
    url = 'https://{0}.chiwawa.one/api/public/v1/groups/{1}/messages'.format(companyId, groupId)
    headers = {
        'Content-Type': 'application/json',
        'X-Chiwawa-API-Token': env['CHIWAWA_API_TOKEN']
    }
    content = {
        'text': message
    }
    requests.post(url, headers=headers, data=json.dumps(content))
    return


def main(mode=MODE, target_text='こんにちは。世界。これはテストのメッセージです。お疲れ様です。'):
    print('SUMMARY MODE: ', MODE_SUMMARY)
    print('SENRYU MODE:', mode)
    print('TARGET TEXT', target_text)
    words = summarize(target_text, method=MODE_SUMMARY)
    print('CREATED WORDS', words)

    if mode == 'S':
        created_senryu = senryu(
            words=words, 
            mode=mode, 
            encoder_model=encoder_model_shimizu, 
            decoder_model=decoder_model_shimizu, 
            is_load_weight=True, 
            word2vec_model=word2vec_model, 
            tokenizer=tokenizer, 
            detokenizer=detokenizer
            )
    elif mode == 'W':
        created_senryu = senryu(
            words=words, 
            mode=mode, 
            encoder_model=encoder_model_watanabe, 
            decoder_model=decoder_model_watanabe, 
            is_load_weight=True, 
            word2vec_model=word2vec_model, 
            tokenizer=tokenizer, 
            detokenizer=detokenizer
            )
    
    # print('CREATED SENRYU', created_senryu)
    print('CREATED SENRYU', created_senryu)
    return created_senryu


if __name__ == "__main__":
    pass
