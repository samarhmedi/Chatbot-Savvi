import random
import nltk
import numpy as np

lemmatizer = nltk.WordNetLemmatizer()
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import json
import pickle

nltk.download('wordnet')
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']
data_file = open('intents.json').read()
intents = json.loads(data_file)

# token for every intent we run every pattern we append it to a tuple
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize here
        w = nltk.word_tokenize(pattern)
        #print('Token is: {}'.format(w))
        words.extend(w)
        #it looks something like (['hey','you'],'greetings)
        documents.append((w, intent['tag']))
        #add the tag to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print('Words list is: {}'.format(words))
# print('Docs are: {}'.format(documents))
# print('classes are: {}'.format(classes))
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# remove duplicates set are not allowed to have duplicates then back to lists
words = list(set(words))
classes = list(set(classes))
# print(words)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# creating an empty array
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    print('Current pattern Words:{}'.format(pattern_words))
    # creating a bag so if the word is in our pattern words we will have 1 if not 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # print('current bag: {}'.format(bag))

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    # print('Current output: {}'.format(output_row))
    # so the x is the bag and the y is the output_row
    training.append([bag, output_row])
    # print('Training: {}'.format(training))
# undo the sequenced data by shuffling because its best to avoid patterns of data so that our model doesn't learn the data
random.shuffle(training)
# converting to numpy array ( x array and y array )
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# print('x: {}'.format(train_x))
# print('y: {}'.format(train_y))
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# randomly drop a layer
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compile the model and define an optimizer function (SGD)
# lr learning rate , decay reducing lr , momentum speed of training
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
mfit = model.fit(np.array(train_x), np.array(train_y), epochs=240, batch_size=5, verbose=1)
model.save('savvi_model.h5', mfit)
print('SAVVI ALIVE!')
