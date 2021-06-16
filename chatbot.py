import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow 
import random
import json
import pickle


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words,labels,training,output = pickle.load(f)

except:
    words = []  #contains all the stemmed words from all the patterns
    labels = [] #contains all the tags
    doc_x = []  #stores the patterns
    doc_y = []  #stores the tags for the corresponding pattern
    #doc_x and doc_y are important for training the data.

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  
            words.extend(wrds)      #extending the entire list instead of adding word by word from the list.
            doc_x.append(wrds)   #each entry in doc_x correseponds to the entry in doc_y. For training the data.
            doc_y.append(intent["tag"])

        if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]   #stemming to get the root word
    words = sorted(list(set(words)))    #remove the duplicates
    labels = sorted(labels)


    #Preprocessing the data
    #neural networks only understand numbers and not strings
    #hence we one-hot encode the data 

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]  #put 'length of items in tags' many zeros.

    for x, doc in enumerate(doc_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc] #stemming the words from patterns again

        for w in words:      
            if w in wrds:
                bag.append(1)       #if the word exists, we put 1 else 0
            else:
                bag.append(0)
        
            output_row = out_empty[:]
            output_row[labels.index(doc_y[x])] = 1

            training.append(bag)
            output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words,labels,training,output),f)

#training the model

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)    #two hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:        
    model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg["responses"]
        
        print(random.choice(responses))

chat()