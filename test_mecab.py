import MeCab

tagger = MeCab.Tagger("")
print(tagger.parse("今日は最高の気分です"))
