import os
import pandas as pd
import tensorflow.keras.backend as K
from gensim.corpora import Dictionary
from keras.preprocessing.sequence import pad_sequences
from hazm import Normalizer, POSTagger, word_tokenize, Lemmatizer, stopwords_list
from parsivar import FindStems
import numpy as np
from keras.layers import Input, Dense, Embedding, Dropout
from keras.models import Model
from keras.layers import BatchNormalization 
from keras import regularizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense
# Set Directories 

pardir = os.path.dirname(os.getcwd())
PATH = os.path.join(pardir,"data\\caption_all_fa\\")
os.chdir(PATH)
SAVED = os.path.join(pardir,"data\\saved\\")
# Prepare for data

def load_doc(filename):
    file = open(file=filename, mode='r', encoding="utf-8")
    text = file.read()
    file.close()
    return text

# Map filename to image,text,label for train,evaluation and test
def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = ' '.join(image_desc)
            descriptions[image_id].append(desc)
    return descriptions

def load_clean_class(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1]
        if image_desc in dataset:
            if image_desc not in descriptions:
                descriptions[image_desc] = list()
            descriptions[image_desc].append(image_id)
    return descriptions

# OneHot
def load_class_dummy(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1]
        if image_desc in dataset:
            if image_desc not in descriptions:
                descriptions[image_desc] = list()
            image_id=dummies_dict[image_id]
            descriptions[image_desc].append(image_id)
    return descriptions

### Class to one_hot_vector dictionary ###
folder_names = []

for entry_name in os.listdir(PATH):
    entry_path = os.path.join(PATH, entry_name)
    if os.path.isdir(entry_path):
        folder_names.append(entry_name)
folder_names =sorted(folder_names)

dummies = pd.get_dummies(folder_names)
dummies_list=dummies.values.tolist()
dummies_dict=dict(zip(folder_names,dummies_list))
### Text and label dictionary for train ###
filename = SAVED+'/train_image.txt'
train=[]
with open(file=filename, encoding='utf-8', mode='r') as f:
    for line in f.read().splitlines():
        train=train+[line.split(',')[0][:-4]]

train_descriptions = load_clean_descriptions(SAVED+'flower_text_tagged_fa.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

train_class = load_clean_class(SAVED+'flower_class.txt', train)
print('Descriptions: train=%d' % len(train_class))

class_dummy=load_class_dummy(SAVED+'flower_class.txt', train)

### Text and label dictionary for EVAL ###
filename = SAVED+'/val_image.txt'
val=[]
with open(filename, encoding='utf-8',mode='r') as f:
    for line in f.read().splitlines():
        val=val+[line.split(',')[0][:-4]]

val_descriptions = load_clean_descriptions(SAVED+'flower_text_tagged_fa.txt', val)
print('Descriptions: val=%d' % len(val_descriptions))

val_class = load_clean_class(SAVED+'flower_class.txt', val)
print('Descriptions: val=%d' % len(val_class))
val_class_dummy=load_class_dummy(SAVED+'flower_class.txt', val)


### Text and label dictionary for Test ###
filename = SAVED+'/test_image.txt'
test=[]
with open(file=filename, encoding='utf-8',mode='r') as f:
    for line in f.read().splitlines():
        test=test+[line.split(',')[0][:-4]]

test_descriptions = load_clean_descriptions(SAVED+'flower_text_tagged_fa.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

test_class = load_clean_class(SAVED+'flower_class.txt', test)
print('Descriptions: test=%d' % len(test_class))
test_class_dummy=load_class_dummy(SAVED+'flower_class.txt', test)
# Load data

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*؟%؛،ًًًََُّ»«_~'''
def punc(myStr):
  no_punct = ""
  for char in myStr:
    if char not in punctuations:
        no_punct = no_punct + char
  return no_punct

# Load morpheme dictionary for flowers
dictionary2=Dictionary.load(SAVED+"dictionary/flower_dictionary2_fa_lemAndStemWithStopWords")

# Text loading and automatic preprocessing function
MAX_SEQUENCE_LENGTH=300
c=str()

normalizer = Normalizer(correct_spacing=True)
lemmatizer = Lemmatizer(joined_verb_parts=False)
tagger = POSTagger(model=SAVED+'pos_tagger.model')
stop_words = stopwords_list()
my_stemmer = FindStems()

def create_phrase(train_descriptions, class_dummy, MAX_SEQUENCE_LENGTH):

    X_train, y_class = list(), list()

    for key, desc_list in train_descriptions.items():
        c=list()
        for desc in desc_list:

            seq = punc(desc)
            normalized_text = normalizer.normalize(seq)
            tokens = word_tokenize(normalized_text)

            newTokens = []

            tokenAndTag = tagger.tag(tokens)
            newTokenAndTag = []
            for TAT in tokenAndTag:
              newTokenAndTag.append([TAT[0], TAT[1].replace('PRON', 'PRO').replace('ADJ', 'AJ').replace('VERB', 'V').split(',')[0]])
            tokenAndTag = newTokenAndTag

            j = 0
            for token in tokens :
              for lemToken in lemmatizer.lemmatize(word=token, pos=tokenAndTag[j][1]).split('#') :
                  if lemToken != '' :#if lemToken not in stop_words and lemToken != ''  :
                    newTokens.append(lemToken)
              for stemToken in my_stemmer.convert_to_stem(token).split('&') :
                  if  stemToken != ''  :#if stemToken not in stop_words and stemToken != ''  :
                    newTokens.append(stemToken)
              j += 1
            tokens = newTokens
            tokens = [token for token in tokens if token != '']

            b = tagger.tag(tokens)

            aaa=[]
            for y in b:
                aaa.append(y[0])

            c+=aaa
        d=dictionary2.doc2idx(c)
        X_train.append(np.array(d))
        if key in class_dummy:
            y_class.append(np.array(class_dummy[key][0]))

    X_train=pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    return np.array(X_train),np.array(y_class)
# Import text into training, validation, and test sets
X_train, y_train = create_phrase(train_descriptions, class_dummy,MAX_SEQUENCE_LENGTH)
X_val, y_val = create_phrase(val_descriptions, val_class_dummy,MAX_SEQUENCE_LENGTH)
X_test, y_test = create_phrase(test_descriptions, test_class_dummy,MAX_SEQUENCE_LENGTH)
# Load pre-trained embedding matrix
EMBEDDING_DIM=300
word_index=dictionary2.token2id

embedding_path = SAVED + 'skipgram/skigram1_fa_new_300_lemStemWithStopWords.txt'

embeddings_index = dict()
f = open(file = embedding_path, encoding= "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#LSTM

def text_LSTM(DROP_OUT1, DROP_OUT2, DENSE_NUM, LAMBDA1, LAMBDA2):
    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embedded_sequences = embedding_layer(sequence_input)
    
    # LSTM layer
    lstm = LSTM(128, return_sequences=True)(embedded_sequences)
    lstm = LSTM(64)(lstm)
    dropout = Dropout(DROP_OUT1)(lstm)

    dense = Dense(DENSE_NUM, activation='relu')(dropout)
    
    dense = BatchNormalization()(dense)

    dropout = Dropout(DROP_OUT2)(dense)

    output = Dense(units=102, activation='softmax', kernel_regularizer=regularizers.L1L2(l1=LAMBDA1, l2=LAMBDA2), kernel_initializer='he_normal')(dropout)

    model = Model(sequence_input, output)

    return model

BATCH_SIZE = 128
EPOCHS = 80
DROP_OUT1 = 0.001
DROP_OUT2 = 0.005
LAMBDA1 = 0.01
LAMBDA2 = 0.05
DENSE_NUM = 128

model = text_LSTM(DROP_OUT1, DROP_OUT2, DENSE_NUM, LAMBDA1, LAMBDA2)


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

initial_learning_rate = 0.0025

rmsOpt = RMSprop(learning_rate = 0.002, weight_decay=0.00025, use_ema=True)

model.compile(loss='categorical_crossentropy', optimizer=rmsOpt, metrics=['accuracy'])

print(BATCH_SIZE,
    EPOCHS,
    DROP_OUT1,
    DROP_OUT2,
    LAMBDA1,
    LAMBDA2,
    DENSE_NUM)

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], validation_data=(X_val, y_val), shuffle=True, validation_batch_size=BATCH_SIZE*4)

score = model.evaluate(X_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100)) 