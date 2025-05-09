import os,re,sys,pickle
import pandas as pd
import tensorflow.keras.backend as K
from gensim.corpora import Dictionary
from keras.preprocessing.sequence import pad_sequences
from hazm import Normalizer, POSTagger, word_tokenize, Lemmatizer, stopwords_list
from parsivar import FindStems
import numpy as np
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.layers import BatchNormalization
from keras import regularizers
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
# Set Directories 

pardir = os.path.dirname(os.getcwd())
PATH = os.path.join(pardir,"data\\caption_all_fa\\")
os.chdir(PATH)
SAVED = os.path.join(pardir,"data\\saved\\")

# Prepare for data #

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

# This function is to load images features
def load_photo_features(filename, dataset):
	all_features = pickle.load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features

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

### Image, text, label dictionary for train ###
filename = SAVED+'/train_image.txt'
train=[]
with open(file=filename, encoding='utf-8', mode='r') as f:
    for line in f.read().splitlines():
        train=train+[line.split(',')[0][:-4]]

train_descriptions = load_clean_descriptions(SAVED+'flower_text_tagged_fa.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

with open(SAVED+"train_image_features_ENetB2.pkl","rb") as f:
    train_features = pickle.load(f)
# train_features = load_photo_features(SAVED+'flower_image_features.pkl', train)
print('Photos: train=%d' % len(train_features))

train_class = load_clean_class(SAVED+'flower_class.txt', train)
print('Descriptions: train=%d' % len(train_class))

class_dummy=load_class_dummy(SAVED+'flower_class.txt', train)

### Image, text, label dictionary for EVAL ###
filename = SAVED+'/val_image.txt'
val=[]
with open(filename, encoding='utf-8',mode='r') as f:
    for line in f.read().splitlines():
        val=val+[line.split(',')[0][:-4]]

val_descriptions = load_clean_descriptions(SAVED+'flower_text_tagged_fa.txt', val)
print('Descriptions: val=%d' % len(val_descriptions))

with open(SAVED+"eval_image_features_ENetB2.pkl","rb") as f:
    val_features =pickle.load(f)
#val_features = load_photo_features(SAVED+'flower_image_features.pkl', val)
print('Photos: vak=%d' % len(val_features))

val_class = load_clean_class(SAVED+'flower_class.txt', val)
print('Descriptions: train=%d' % len(val_class))
val_class_dummy=load_class_dummy(SAVED+'flower_class.txt', val)


### Image, text, label dictionary for Test ###
filename = SAVED+'/test_image.txt'
test=[]
with open(file=filename, encoding='utf-8',mode='r') as f:
    for line in f.read().splitlines():
        test=test+[line.split(',')[0][:-4]]


test_descriptions = load_clean_descriptions(SAVED+'flower_text_tagged_fa.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

with open(SAVED+"test_image_features_ENetB2.pkl","rb") as f:
    test_features =pickle.load(f)
#test_features = load_photo_features(SAVED+'flower_image_features.pkl', test)
print('Photos: test=%d' % len(test_features))

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

# Load dictionary for flowers
dictionary2=Dictionary.load(SAVED+"dictionary/flower_dictionary2_fa_lemAndStemWithStopWords")

# Image and text loading and automatic preprocessing function
MAX_SEQUENCE_LENGTH=300
c=str()

normalizer = Normalizer(correct_spacing=True)
lemmatizer = Lemmatizer(joined_verb_parts=False)
tagger = POSTagger(model=SAVED+'pos_tagger.model')
stop_words = stopwords_list()
my_stemmer = FindStems()

def create_phrase(train_features, train_descriptions, class_dummy,MAX_SEQUENCE_LENGTH):

    X_image, X_text, y_class = list(), list(), list()
    #tweet=Okt()

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
        X_text.append(np.array(d))
        if key in class_dummy:
            y_class.append(np.array(class_dummy[key][0]))
        X_image.append(train_features[key+".jpg"][0])

    X_text=pad_sequences(X_text, maxlen=MAX_SEQUENCE_LENGTH)
    return np.array(X_text),np.array(X_image),np.array(y_class)
# Import images and text into training, validation, and test sets
X_text, X_image, y_class =create_phrase(train_features, train_descriptions, class_dummy,MAX_SEQUENCE_LENGTH)
X_text_val, X_image_val, y_class_val =create_phrase(val_features, val_descriptions, val_class_dummy,MAX_SEQUENCE_LENGTH)
X_text_test, X_image_test, y_class_test =create_phrase(test_features, test_descriptions, test_class_dummy,MAX_SEQUENCE_LENGTH)

# Load pre-trained embedding matrix
EMBEDDING_DIM=300  #skipgram Dim
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
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#Model definition

def Modified_m_CNN(DROP_OUT,DROP_OUT2,LAMBDA):
	inputs1 = Input(shape=(4096,))
	x= Reshape((16,1,256))(inputs1)
	x=BatchNormalization()(x)
	conv_x=Conv2D(256, kernel_size=(14,1), padding='valid', kernel_initializer='he_normal', activation='relu')(x)

	conv_x = Dropout(DROP_OUT2)(conv_x)
	Max_x = MaxPool2D(pool_size=(2,1))(conv_x)
	inputs2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
	y = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True) (inputs2)
	reshape = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(y)
	conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), padding='valid', kernel_initializer='he_normal', activation='relu')(reshape)
	conv_0=BatchNormalization()(conv_0)
	conv_0 = Dropout(DROP_OUT2)(conv_0)
	conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), padding='valid', kernel_initializer='he_normal', activation='relu')(reshape)
	conv_1 =BatchNormalization()(conv_1 )
	conv_1 = Dropout(DROP_OUT2)(conv_1)
	conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), padding='valid', kernel_initializer='he_normal', activation='relu')(reshape)
	conv_2 =BatchNormalization()(conv_2 )
	conv_2 = Dropout(DROP_OUT2)(conv_2)
	maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
	maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
	maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
	concat1 = Concatenate(axis=1)([maxpool_0,Max_x])
	concat2 = Concatenate(axis=1)([maxpool_1,Max_x])
	concat3 = Concatenate(axis=1)([maxpool_2,Max_x])
	concat4 = Concatenate(axis=1)([concat1,concat2 ,concat3 ])
	a=Conv2D(512, kernel_size=(5,1), padding='valid', kernel_initializer='he_normal', activation='relu')(concat4)
	a=BatchNormalization()(a)
	a=Dropout(DROP_OUT2)(a)
	a=MaxPool2D(pool_size=(2,1))(a)
	a = Flatten()(a)
	z = Dropout(DROP_OUT)(a)
	output = Dense(units=102, activation='softmax',kernel_regularizer=regularizers.l2(LAMBDA), kernel_initializer='he_normal')(z)
	model = Model(inputs=[inputs1, inputs2], outputs=output)
	#print(model.summary())
	return model

BATCH_SIZE = 128
EPOCHS = 80
LAMBDA=0.05
DROP_OUT=0.2
DROP_OUT2=0.4
filter_sizes = [2,3,4]
num_filters = 256

model=Modified_m_CNN(DROP_OUT,DROP_OUT2,LAMBDA)

adamw_opt = AdamW(learning_rate = 0.0005, weight_decay=0.005, use_ema=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(loss='categorical_crossentropy',optimizer=adamw_opt, metrics=['accuracy'])#Adam


history = model.fit([X_image,X_text], y_class, epochs=EPOCHS,
     batch_size=BATCH_SIZE,validation_data=([X_image_val,X_text_val],y_class_val), callbacks=[early_stopping],validation_batch_size=BATCH_SIZE*4)

score = model.evaluate([X_image_test,X_text_test], y_class_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))