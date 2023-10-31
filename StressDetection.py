from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
import os
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))

main = tkinter.Tk()
main.title("Detection of Employee Stress Using Machine Learning")
main.geometry("1300x1200")

global model
global filename
global tokenizer
global X
global Y
global X_train, X_test, Y_train, Y_test
global XX
word_count = 0
global svm_acc,rf_acc
global model

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir = "Tweets")
    pathlabel.config(text=filename)
    textarea.delete('1.0', END)
    textarea.insert(END,'tweets dataset loaded\n')
    

def preprocess():
    global X
    global Y
    global word_count
    X = []
    Y = []
    textarea.delete('1.0', END)
    train = pd.read_csv(filename,encoding='iso-8859-1')
    word_count = 0
    words = []
    for i in range(len(train)):
        label = train.get_value(i,2,takeable = True)
        tweet = train.get_value(i,1,takeable = True)
        tweet = tweet.lower()
        arr = tweet.split(" ")
        msg = ''
        for k in range(len(arr)):
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg+=word+" "
                if word not in words:
                    words.append(word);
        text = msg.strip()
        X.append(text)
        Y.append(int(label))
    X = np.asarray(X)
    Y = np.asarray(Y)
    word_count = len(words)
    textarea.insert(END,'Total tweets found in dataset : '+str(len(X))+"\n")
    textarea.insert(END,'Total words found in all tweets : '+str(len(words))+"\n\n")
    featureExtraction()

def featureExtraction():
    global X
    global Y
    global XX
    global tokenizer
    global X_train, X_test, Y_train, Y_test
    max_fatures = word_count
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(X)
    XX = tokenizer.texts_to_sequences(X)
    XX = pad_sequences(XX)
    indices = np.arange(XX.shape[0])
    np.random.shuffle(indices)
    XX = XX[indices]
    Y = Y[indices]
    X_train, X_test, Y_train, Y_test = train_test_split(XX,Y, test_size = 0.13, random_state = 42)
    textarea.insert(END,'Total features extracted from tweets are  : '+str(X_train.shape[1])+"\n")
    textarea.insert(END,'Total splitted records used for training : '+str(len(X_train))+"\n")
    textarea.insert(END,'Total splitted records used for testing : '+str(len(X_test))+"\n") 

def SVM():
    textarea.delete('1.0', END)
    global svm_acc
    rfc = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
    rfc.fit(X_train, Y_train)
    textarea.insert(END,"SVM Prediction Results\n") 
    prediction_data = rfc.predict(X_test) 
    svm_acc = accuracy_score(Y_test,prediction_data)*100
    textarea.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n\n")
    
def RandomForest():
    global rf_acc
    global model
    rfc = RandomForestClassifier(n_estimators=20, random_state=0)
    rfc.fit(X_train, Y_train)
    textarea.insert(END,"Random Forest Prediction Results\n") 
    prediction_data = rfc.predict(X_test) 
    rf_acc = accuracy_score(Y_test,prediction_data)*100
    textarea.insert(END,"Random Forest Accuracy : "+str(rf_acc)+"\n")
    model = rfc

def predict():
    textarea.delete('1.0', END)
    testfile = filedialog.askopenfilename(initialdir = "Tweets")
    test = pd.read_csv(testfile,encoding='iso-8859-1')

    for i in range(len(test)):
        tweet = test.get_value(i,0,takeable = True)
        arr = tweet.split(" ")
        msg = ''
        for j in range(len(arr)):
            word = arr[j].strip()
            if len(word) > 2 and word not in stop_words:
                msg+=word+" "
        text = msg.strip()
        mytext = [text]
        twts = tokenizer.texts_to_sequences(mytext)
        twts = pad_sequences(twts, maxlen=83, dtype='int32', value=0)
        stress = model.predict(twts)
        print(stress)
        if stress == 0:
            textarea.insert(END,text+' === Prediction Result : Not Stressed\n\n')
        if stress == 1:
            textarea.insert(END,text+' === Prediction Result : Stressed\n\n')
        

def graph():
    height = [svm_acc,rf_acc]
    bars = ('SVM ACC','Random Forest ACC')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()      

font = ('times', 16, 'bold')
title = Label(main, text='Detection of Employee Stress Using Machine Learning')
title.config(bg='yellow green', fg='saddle brown')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Tweets Dataset", command=upload)
upload.place(x=780,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='royal blue', fg='rosy brown')  
pathlabel.config(font=font1)           
pathlabel.place(x=780,y=150)

preprocessButton = Button(main, text="Data Preprocessing & Features Extraction", command=preprocess)
preprocessButton.place(x=780,y=200)
preprocessButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=SVM)
svmButton.place(x=780,y=250)
svmButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest Algorithm", command=RandomForest)
rfButton.place(x=780,y=300)
rfButton.config(font=font1)

classifyButton = Button(main, text="Predict Stress", command=predict)
classifyButton.place(x=780,y=350)
classifyButton.config(font=font1)

modelButton = Button(main, text="Accuracy Graph", command=graph)
modelButton.place(x=780,y=400)
modelButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=30,width=90)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=100)
textarea.config(font=font1)


main.config(bg='cadet blue')
main.mainloop()
