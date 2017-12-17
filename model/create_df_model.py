import json
import corenlp
import copy
import pandas as pd

# 学習データ
LEARNING_DOCUMENT = "../data/train.csv"

# パーサの生成
CORENLP_DIR = "../stanford-corenlp-full-2014-08-27"
parser = corenlp.StanfordCoreNLP(corenlp_path=CORENLP_DIR)


class CreateDfModel:
    
    #コンストラクタ
    def __init__(self):
        #{学習データ内の単語:その単語が含まれる文数}
        #【例】
        #{りんご:18}だったら、学習データ(STS_train_feature.txt)の中に
        #りんごという単語が存在する文数は18文ある。
        self.__sentence_count_df = pd.DataFrame([], index=list('0'),columns=list())
        #一文に出てきた単語の原型の集合
        self.__add_sentence_count = set()

    # 学習データ(STS_train_feature.txt)における
    # ある単語が含まれる文数をカウント
    def __count_sentence_in_learning_document(self,add_sentence_count):
        for word in add_sentence_count:
            if word in self.__sentence_count_df.columns:
                self.__sentence_count_df[word] = self.__sentence_count_df[word] + 1
            else:
                self.__sentence_count_df[word] = 1
        self.__add_sentence_count.clear()


    # 実行関数
    def run(self):
        k = 0
        train_df = pd.read_csv(LEARNING_DOCUMENT)
        for sentence1 in train_df["question1"]:
            sentence_json = json.loads(parser.parse(sentence1))
            for j in range(len(sentence_json['sentences'][0]['words'])):
                # 単語を原型にする。
                try:
                	word = sentence_json['sentences'][0]['words'][j][1]['Lemma']
                except:
                    pass
                #一文に出てきた単語
                #ただし重複なし
                self.__add_sentence_count.add(word)
                self.__count_sentence_in_learning_document(self.__add_sentence_count)
            print(k)
            print(sentence1)
            k = k + 1

        print(k)
        for sentence2 in train_df["question2"]:
            sentence_json = json.loads(parser.parse(sentence2))
            for j in range(len(sentence_json['sentences'][0]['words'])):
                # 単語を原型にする。
               	try:
                	word = sentence_json['sentences'][0]['words'][j][1]['Lemma']
                except:
                    pass
                #一文に出てきた単語
                #ただし重複なし
                self.__add_sentence_count.add(word)
                self.__count_sentence_in_learning_document(self.__add_sentence_count)

        print(k)
        k = k + 1
        self.__sentence_count_df.to_csv("../data/STS_train_df.csv")


if __name__ == '__main__':
    create_df_model = CreateDfModel()
    create_df_model.run()


        