import os, pickle
import numpy as np
import pandas as pd
from keras.layers import Input, Conv2D, MaxPool2D, Reshape, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# Set Directories 

pardir = os.path.dirname(os.getcwd())
PATH = os.path.join(pardir,"data\\caption_all_fa\\")
os.chdir(PATH)
SAVED = os.path.join(pardir,"data\\saved\\")
# Load data functions

def load_doc(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        text = file.read()
    return text

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

def load_class_dummy(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1]
        if image_desc in dataset:
            if image_desc not in descriptions:
                descriptions[image_desc] = list()
            image_id = dummies_dict[image_id]
            descriptions[image_desc].append(image_id)
    return descriptions

# Class to one_hot_vector dictionary
folder_names = sorted([entry_name for entry_name in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, entry_name))])
dummies = pd.get_dummies(folder_names)
dummies_list = dummies.values.tolist()
dummies_dict = dict(zip(folder_names, dummies_list))

# Load train data
filename = SAVED + '/train_image.txt'
train = [line.split(',')[0][:-4] for line in open(filename, encoding='utf-8').read().splitlines()]

with open(SAVED + "train_image_features_ENetB2.pkl", "rb") as f: #choose image features
    train_features = pickle.load(f)

train_class = load_clean_class(SAVED + 'flower_class.txt', train)
class_dummy = load_class_dummy(SAVED + 'flower_class.txt', train)

# Load val data
filename = SAVED + '/val_image.txt'
val = [line.split(',')[0][:-4] for line in open(filename, encoding='utf-8').read().splitlines()]

with open(SAVED + "eval_image_features_ENetB2.pkl", "rb") as f: #choose image features
    val_features = pickle.load(f)

val_class = load_clean_class(SAVED + 'flower_class.txt', val)
val_class_dummy = load_class_dummy(SAVED + 'flower_class.txt', val)

# Load test data
filename = SAVED + '/test_image.txt'
test = [line.split(',')[0][:-4] for line in open(filename, encoding='utf-8').read().splitlines()]

with open(SAVED + "test_image_features_ENetB2.pkl", "rb") as f: #choose image features
    test_features = pickle.load(f)

test_class = load_clean_class(SAVED + 'flower_class.txt', test)
test_class_dummy = load_class_dummy(SAVED + 'flower_class.txt', test)

# Prepare data for training
def prepare_data(features, class_dummy):
    X_image, y_class = [], []
    for key in features.keys():
        X_image.append(features[key][0])
        key = key.replace(".jpg", "")
        if key in class_dummy:
            y_class.append(np.array(class_dummy[key][0]))
    return np.array(X_image), np.array(y_class)

X_image, y_class = prepare_data(train_features, class_dummy)
X_image_val, y_class_val = prepare_data(val_features, val_class_dummy)
X_image_test, y_class_test = prepare_data(test_features, test_class_dummy)
# Model definition
def Modified_m_CNN(DROP_OUT, DROP_OUT2, LAMBDA):
    inputs = Input(shape=(4096,))
    x = Reshape((16, 1, 256))(inputs)
    x = BatchNormalization()(x)
    conv_x = Conv2D(256, kernel_size=(14, 1), padding='valid', kernel_initializer='he_normal', activation='relu')(x)
    conv_x = Dropout(DROP_OUT2)(conv_x)
    max_x = MaxPool2D(pool_size=(2, 1))(conv_x)
    x = Flatten()(max_x)
    x = Dropout(DROP_OUT)(x)
    output = Dense(units=102, activation='softmax', kernel_regularizer='l2')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

BATCH_SIZE = 128
EPOCHS = 40
LAMBDA = 0.05
DROP_OUT = 0.2
DROP_OUT2 = 0.4

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = Modified_m_CNN(DROP_OUT, DROP_OUT2, LAMBDA)

# Model detailed settings
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Model fit
history = model.fit(X_image, y_class, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_image_val, y_class_val), shuffle=True, callbacks=[early_stopping])

# Model evaluation
score = model.evaluate(X_image_test, y_class_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))