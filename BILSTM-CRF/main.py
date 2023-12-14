import numpy as np
import tensorflow as tf   #1.2
from tensorflow.contrib import rnn, crf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Function to read the data
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sentences, sentence = [], []
    for line in lines:
        if line != '\n':
            char, label = line.strip().split()
            sentence.append((char, label))
        else:
            sentences.append(sentence)
            sentence = []
    return sentences

# Custom pad_sequences function
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if padding == 'post':
            x[idx, :len(s)] = s
        elif padding == 'pre':
            x[idx, -len(s):] = s
    return x

# Handle unknown words in test set
def handle_unknown_words(sentence, vocab):
    return [w if w in vocab else 'UNK' for w in sentence]


import os
current_path = os.getcwd()
print(current_path)

# Read and preprocess data
train_data = read_data(current_path+'\\data\\train.txt')
test_data = read_data(current_path+'\\data\\test.txt')

# Create lists to store sentences and corresponding labels
train_sentences = [[char for char, label in s] for s in train_data]
train_labels = [[label for char, label in s] for s in train_data]
test_sentences = [[char for char, label in s] for s in test_data]
test_labels = [[label for char, label in s] for s in test_data]

# Label encoding for words and tags
word_encoder = LabelEncoder()
tag_encoder = LabelEncoder()

all_words = [w for s in train_sentences for w in s]
all_tags = [t for s in train_labels for t in s]

word_encoder.fit(all_words)
tag_encoder.fit(all_tags)

# Add 'UNK' to word_encoder classes
vocab = set(word_encoder.classes_)
vocab.add('UNK')
word_encoder.classes_ = np.array(list(vocab))

# Handle unknown words
test_sentences = [handle_unknown_words(s, vocab) for s in test_sentences]

train_sentences_enc = [word_encoder.transform(s) for s in train_sentences]
train_labels_enc = [tag_encoder.transform(s) for s in train_labels]
test_sentences_enc = [word_encoder.transform(s) for s in test_sentences]
test_labels_enc = [tag_encoder.transform(s) for s in test_labels]

# Pad the sequences
train_sentences_padded = pad_sequences(train_sentences_enc, padding='post')
train_labels_padded = pad_sequences(train_labels_enc, padding='post')
# test_sentences_padded = pad_sequences(test_sentences_enc, padding='post')
# test_labels_padded = pad_sequences(test_labels_enc, padding='post')
test_sentences_padded = pad_sequences(test_sentences_enc, maxlen=510, padding='post')
test_labels_padded = pad_sequences(test_labels_enc, maxlen=510, padding='post')

# Hyperparameters
learning_rate = 0.01
training_epochs = 45
batch_size = 64
sequence_length = train_sentences_padded.shape[1]
vocab_size = len(word_encoder.classes_)
n_tags = len(tag_encoder.classes_)
embedding_dim = 8
hidden_units = 96

# TensorFlow Graph input
input_data = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_data')
labels = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='labels')

# Word embedding
word_embeddings = tf.get_variable("word_embeddings", [vocab_size, embedding_dim])
embedded_words = tf.nn.embedding_lookup(word_embeddings, input_data)

# BiLSTM layer
fw_cell = rnn.BasicLSTMCell(hidden_units)
bw_cell = rnn.BasicLSTMCell(hidden_units)
(outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                             cell_bw=bw_cell,
                                                             inputs=embedded_words,
                                                             dtype=tf.float32)
outputs = tf.concat([outputs_fw, outputs_bw], axis=2)

# Fully connected layer
w = tf.get_variable(name="W", dtype=tf.float32, shape=[2 * hidden_units, n_tags])
b = tf.get_variable(name="b", dtype=tf.float32, shape=[n_tags])

outputs = tf.reshape(outputs, [-1, 2 * hidden_units])
pred = tf.matmul(outputs, w) + b
scores = tf.reshape(pred, [-1, sequence_length, n_tags])

# CRF layer
log_likelihood, transition_params = crf.crf_log_likelihood(scores, labels, tf.count_nonzero(input_data, 1))

# Loss and optimizer
loss = tf.reduce_mean(-log_likelihood)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Training the model
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for i in range(0, len(train_sentences_padded), batch_size):
            batch_data = train_sentences_padded[i:i + batch_size]
            batch_labels = train_labels_padded[i:i + batch_size]
            _, loss_value = sess.run([optimizer, loss], feed_dict={input_data: batch_data, labels: batch_labels})

        # print(f"Epoch {epoch + 1}, Loss: {loss_value}")
        print(f"Epoch {epoch + 1}")

    # Testing the model
    test_pred, trans_params = sess.run([scores, transition_params], feed_dict={input_data: test_sentences_padded})

    # Post-process to extract named entities
    y_pred, y_true = [], []
    for i in range(len(test_pred)):
        logits = test_pred[i]
        tags = test_labels_padded[i]
        sequence_lengths = sum(1 for t in tags if t != 0)
        viterbi_seq, _ = crf.viterbi_decode(logits[:sequence_lengths], trans_params)
        y_pred.extend(viterbi_seq)
        y_true.extend(tags[:sequence_lengths])

    # print(classification_report(y_true, y_pred, target_names=tag_encoder.classes_))


# Find the integer encoding for 'O'
o_label = tag_encoder.transform(['O'])[0]

# Create new lists for y_true and y_pred without the 'O' labels
y_true_filtered = [label for label in y_true if label != o_label]
y_pred_filtered = [pred for pred, true in zip(y_pred, y_true) if true != o_label]

# Identify the integer labels corresponding to the remaining classes (excluding 'O')
remaining_labels = [i for i, label in enumerate(tag_encoder.classes_) if label != 'O']

# Generate the classification report for the remaining classes
print(classification_report(y_true_filtered, y_pred_filtered, labels=remaining_labels, target_names=[label for label in tag_encoder.classes_ if label != 'O'], digits=4))
