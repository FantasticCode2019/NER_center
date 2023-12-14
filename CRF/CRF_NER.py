import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
import yaml
import warnings
warnings.filterwarnings('ignore')

def load_data(data_path):
    data_read_all = list()
    data_sent_with_label = list()
    with open(data_path, mode='r', encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                data_read_all.append(data_sent_with_label.copy())
                data_sent_with_label.clear()
            else:
                data_sent_with_label.append(tuple(line.strip().split(" ")))
    return data_read_all

def word2features(sent, i):
    word = sent[i][0]
    #构造特征字典，我这里因为整体句子长度比较长，滑动窗口的大小设置的是6 在特征的构建中主要考虑了字的标识,是否是数字和字周围的特征信息
    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit(),
    }
    #该字的前一个字
    if i > 0:
        word1 = sent[i-1][0]
        words = word1 + word
        features.update({
            '-1:word': word1,
            '-1:words': words,
            '-1:word.isdigit()': word1.isdigit(),
        })
    else:
        #添加开头的标识 BOS(begin of sentence)
        features['BOS'] = True
    #该字的前两个字
    if i > 1:
        word2 = sent[i-2][0]
        word1 = sent[i-1][0]
        words = word1 + word2 + word
        features.update({
            '-2:word': word2,
            '-2:words': words,
            '-3:word.isdigit()': word2.isdigit(),
        })
    #该字的前三个字
    if i > 2:
        word3 = sent[i - 3][0]
        word2 = sent[i - 2][0]
        word1 = sent[i - 1][0]
        words = word1 + word2 + word3 + word
        features.update({
            '-3:word': word3,
            '-3:words': words,
            '-3:word.isdigit()': word3.isdigit(),
        })
    #该字的后一个字
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        words = word1 + word
        features.update({
            '+1:word': word1,
            '+1:words': words,
            '+1:word.isdigit()': word1.isdigit(),
        })
    else:
    #若改字为句子的结尾添加对应的标识end of sentence
        features['EOS'] = True
    #该字的后两个字
    if i < len(sent)-2:
        word2 = sent[i + 2][0]
        word1 = sent[i + 1][0]
        words = word + word1 + word2
        features.update({
            '+2:word': word2,
            '+2:words': words,
            '+2:word.isdigit()': word2.isdigit(),
        })
    #该字的后三个字
    if i < len(sent)-3:
        word3 = sent[i + 3][0]
        word2 = sent[i + 2][0]
        word1 = sent[i + 1][0]
        words = word + word1 + word2 + word3
        features.update({
            '+3:word': word3,
            '+3:words': words,
            '+3:word.isdigit()': word3.isdigit(),
        })
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [ele[-1] for ele in sent]

train=load_data('train.txt')
valid=load_data('test.txt')
print('训练集规模:',len(train))
print('验证集规模:',len(valid))
sample_text=''.join([c[0] for c in train[0]])
sample_tags=[c[1] for c in train[0]]
X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]
X_dev = [sent2features(s) for s in valid]
y_dev = [sent2labels(s) for s in valid]
print(X_train[0])

crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.8,c2=0.018,max_iterations=9,
                                 all_possible_transitions=True,verbose=True)
crf_model.fit(X_train, y_train)

labels=list(crf_model.classes_)
labels.remove("O")  #对于O标签的预测我们不关心，就直接去掉
y_pred = crf_model.predict(X_dev)
metrics.flat_f1_score(y_dev, y_pred,
                      average='weighted', labels=labels)

precision = metrics.flat_precision_score(y_dev, y_pred,average='weighted', labels=labels)
recall = metrics.flat_recall_score(y_dev, y_pred,average='weighted', labels=labels)
f1_score = metrics.flat_f1_score(y_dev, y_pred,average='weighted', labels=labels)

# print(precision,recall,f1_score)

from sklearn.metrics import classification_report

# 如果你的 y_dev 和 y_pred 是嵌套结构，你需要将它们扁平化
y_true_flat = [item for sublist in y_dev for item in sublist]
y_pred_flat = [item for sublist in y_pred for item in sublist]
def unify_labels(label):
    for unified_label, original_labels in unified_labels_mapping.items():
        if label in original_labels:
            return unified_label
    return label

unified_labels_mapping = {
    'BehavioralHazard': ['B-BehavioralHazard', 'I-BehavioralHazard'],
    'RumorObject': ['B-RumorObject', 'I-RumorObject'],
    'RumorSubject': ['B-RumorSubject', 'I-RumorSubject'],
    'RumorType': ['B-RumorType', 'I-RumorType'],
    'UnlawfulAct': ['B-UnlawfulAct', 'I-UnlawfulAct'],
}
# unified_labels_mapping = {
#     'PER': ['B-PER', 'I-PER'],
#     'LOC': ['B-LOC', 'I-LOC'],
#     'ORG': ['B-ORG', 'I-ORG'],
# }
y_true_unified = [unify_labels(label) for label in y_true_flat]
y_pred_unified = [unify_labels(label) for label in y_pred_flat]

report = classification_report(y_true_unified, y_pred_unified, labels=list(unified_labels_mapping.keys()), digits=3)
print(report)

