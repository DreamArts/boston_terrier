# encoding: utf8

from __future__ import unicode_literals
import re
from bs4 import BeautifulSoup
import unicodedata

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def other_processings(s):
    s = s.lower()  # アルファベットの大文字を全て小文字にする
    s = re.sub(r'http\S+', '', s)  # urlを除去
    # 改行を半角空白スペースで置換
    s = s.splitlines()
    s = ' '.join(s)
    s = s.replace('\t', ' ')
    #  HTMLとJSのタグを除去
    soup = BeautifulSoup(s, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    s = soup.get_text()
    s = ''.join(s.splitlines())
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    s = other_processings(s)
    return s

if __name__ == "__main__":
    assert "0123456789" == normalize_neologd("０１２３４５６７８９")
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" == normalize_neologd("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ")
    assert "abcdefghijklmnopqrstuvwxyz" == normalize_neologd("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ")
    assert "!\"#$%&'()*+,-./:;<>?@[¥]^_`{|}" == normalize_neologd("！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝")
    assert "＝。、・「」" == normalize_neologd("＝。、・「」")
    assert "ハンカク" == normalize_neologd("ﾊﾝｶｸ")
    assert "o-o" == normalize_neologd("o₋o")
    assert "majikaー" == normalize_neologd("majika━")
    assert "わい" == normalize_neologd("わ〰い")
    assert "スーパー" == normalize_neologd("スーパーーーー")
    assert "!#" == normalize_neologd("!#")
    assert "ゼンカクスペース" == normalize_neologd("ゼンカク　スペース")
    assert "おお" == normalize_neologd("お             お")
    assert "おお" == normalize_neologd("      おお")
    assert "おお" == normalize_neologd("おお      ")
    assert "検索エンジン自作入門を買いました!!!" == \
        normalize_neologd("検索 エンジン 自作 入門 を 買い ました!!!")
    assert "アルゴリズムC" == normalize_neologd("アルゴリズム C")
    assert "PRML副読本" == normalize_neologd("　　　ＰＲＭＬ　　副　読　本　　　")
    assert "Coding the Matrix" == normalize_neologd("Coding the Matrix")
    assert "南アルプスの天然水Sparking Lemonレモン一絞り" == \
        normalize_neologd("南アルプスの　天然水　Ｓｐａｒｋｉｎｇ　Ｌｅｍｏｎ　レモン一絞り")
    assert "南アルプスの天然水-Sparking*Lemon+レモン一絞り" == \
        normalize_neologd("南アルプスの　天然水-　Ｓｐａｒｋｉｎｇ*　Ｌｅｍｏｎ+　レモン一絞り")