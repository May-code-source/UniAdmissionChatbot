import random
import pickle
import numpy as np
import json

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intent.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
chatbot_model = load_model('chatbot_model.h5')

def cleaning_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = cleaning_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def class_prediction(sentence):
    bow = bag_of_words(sentence)
    res = chatbot_model.predict(np.array([bow]))[0]
    ERR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    intent_prob_list = []
    for r in results:
        intent_prob_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return intent_prob_list

def getbotResponse(ints, intents_json):
    tag = ints[0]['intent']
    intents_list = intents_json['intents']
    for intent in intents_list:
        if intent['tag'] == tag:
            outcome = random.choice(intent['responses'])
            break
    return outcome

print("Welcome to Solent University UniAdmission Bot Service. Say hello to chat")

while True:
    message = input("")
    target = class_prediction(message)
    result = getbotResponse(target, intents)
    print(result)

