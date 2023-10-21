#Importing relevant libraries
import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

nltk.download("omw-1.4")
nltk.download ("wordnet")
nltk.download ("punkt")
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

#initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

#loading dataset
intents = json.loads(open('intent.json').read())

words = [] #for bag of words model (vocabulary for patterns)
classes = [] #for bag of words model (vocabulary for tags)
docs = [] #
ignore_words = ['?', '!', '.', ','] 

#iterating over the all the intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        list_of_words = nltk.word_tokenize(pattern)
        words.extend(list_of_words)
        docs.append((list_of_words, intent['tag']))
        #adding tag to the classes if it's not there already
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lemmatizer all the words in the vocab and convert them to lowercase
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]

#sorting the vocab and classes in alphabetically order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#convert
training = []
empty_output = [0] * len(classes)



for doc in docs:

    bag = []

    pattern = doc[0]

    pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]

    for word in words:
        bag.append(1) if word in pattern else bag.append(0)

    output_row = list(empty_output)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

chatbot_model = Sequential()
chatbot_model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
chatbot_model.add(Dropout(0.5))
chatbot_model.add(Dense(64, activation='relu'))
chatbot_model.add(Dropout(0.5))
chatbot_model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
chatbot_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


hist = chatbot_model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

chatbot_model.save('chatbot_model.h5', hist)
print("Done!")


