import json
import sys
import math as mt
import pandas as pd
sys.path.append('../')

from model import create_df_model
import numpy as np
import pandas as pd
import corenlp
from collections import OrderedDict

# パーサの生成
CORENLP_DIR = "../stanford-corenlp-full-2014-08-27"
parser = corenlp.StanfordCoreNLP(corenlp_path=CORENLP_DIR)

#文数
SENTENCE_COUNT = 1500

#入力テストデータ
INPUT_TEST_DOCUMENT = "../data/STS_test_feature.txt"

#予測データ
OUTPUT_TEST_DOCUMENT = "../validation/STS_output_label.csv"

#学習データ(STS_test_feature.txt)の一文に含まれる単語数。
#（例）
# word_count = {"りんご":17}
# の場合、学習データの中の一文の中にはりんごが17個はいっている。
# 後に、tfの計算に利用する。
word_count = {}


def open_document(document_name):
	return open(document_name,'r')

def write_file(document_name,scores):
	output_dic = OrderedDict()
	output_dic['test_id'] = [w for w in range(len(scores))]
	output_dic['is_duplicate'] = scores
	output_df = pd.DataFrame(output_dic)
	output_df.to_csv("output.csv",index = False)

#テストデータ(STS_test_feature.txt)の単語数をカウント
#word_count
def count_word_in_test_document(word):
	global word_count
	if word in word_count:
		word_count[word] = word_count[word] + 1
	else:
		word_count[word] = 1

def calculate_tf_idf(word):
	return calculate_tf(word)*calculate_idf(word)

def calculate_tf(word):
	return word_count[word]
	
def calculate_idf(word):
	if word in train_df.columns:
		return mt.log(SENTENCE_COUNT/train_df[word])
	else:
		return mt.log(SENTENCE_COUNT)

def run():
	global word_count
	global train_df
	tfidf_df = []
	k = 0 
	f = open_document(INPUT_TEST_DOCUMENT)
	for line in f:
		df0 = pd.DataFrame([], index=list('0'),columns=list())
		df1 = pd.DataFrame([], index=list('1'),columns=list())
		df = [df0,df1]
		sentence = line[:-1].split('\t')
		for i in range(0,2):
			sentence_json = json.loads(parser.parse(sentence[i]))
			for j in range(len(sentence_json['sentences'][0]['words'])):
				#単語を原型にする。
				word = sentence_json['sentences'][0]['words'][j][1]['Lemma']
				#テストデータの中の単語数をカウント（tfidfのtfの計算に利用する。）
				count_word_in_test_document(word)
			for word in word_count:
				df[i][word] = calculate_tf_idf(word)
			word_count.clear()
		tfidf_df = df[0].append(df[1])
		# if "a" in tfidf_df.columns:
		# 	tfidf_df = tfidf_df.drop("a",axis = 1)

		tfidf_df = tfidf_df.replace(np.nan,0)
		# if k < 10:
		# 	tfidf_df.to_csv("{}.csv".format(k))
		# 	k = k + 1

		cos = np.dot(tfidf_df[0:1].values,tfidf_df[1:2].values.T)/(np.linalg.norm(tfidf_df[0:1].values)*(np.linalg.norm(tfidf_df[1:2])))
		tfidf_score.append(cos[0,0]*5)
	write_file(OUTPUT_TEST_DOCUMENT,tfidf_score)
		
if __name__ == '__main__':
	train_df = pd.read_csv("../data/STS_train_df.csv")
	run()
	
	
	

			