# -*- coding: utf-8 -*-

# First we import the our libraries"Modules"

import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

#real = pd.read_csv('/content/gdrive/My Drive/fake-news/train_ArNews_1.tsv',sep='\t')
real = pd.read_csv('/datasetes/ar/True_Ar.csv',encoding="utf_8")
fake = pd.read_csv('/datasetes/ar/Fake_Ar.csv',encoding="utf_8")

real.shape
real.head

fake_drop = fake.drop(fake.loc[fake.text == ' '].index)
real_drop = real.drop(real.loc[real.text == ' '].index)

# Give labels to data before combining
fake_drop['fake'] = 1
real['fake'] = 0
real.head

combined = pd.concat([fake_drop, real])

## train/test split the text data and labels
features = combined['text']
labels = combined['fake']
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state = 42)

## not removing stop words to maintain word context
max_words = 2000
max_len = 400

token = Tokenizer(num_words=max_words, lower=True, split=' ')
token.fit_on_texts(X_train.values)
sequences = token.texts_to_sequences(X_train.values)
train_sequences_padded = pad_sequences(sequences, maxlen=max_len)

embed_dim = 50
batch_size = 32

model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length = max_len))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(20, return_sequences=True))
model.add(LSTM(20))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

history = model.fit(train_sequences_padded, y_train, batch_size=batch_size, epochs = 5, validation_split=0.2)

# now compare to test values
test_sequences = token.texts_to_sequences(X_test)
test_sequences_padded = pad_sequences(test_sequences, maxlen=max_len)

model.evaluate(test_sequences_padded, y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.legend(['Training', 'Validation'])
plt.ylabel('Accuracy (%)')
plt.xlabel('Epochs')
plt.xticks([0,1,2,3,4])

# can it generalize?
test1 = ["""
وكثفت الحكومة اجتماعاتها في الايام الاخيرة لايجاد موارد مالية اضافية مع مواصلتها التقديمات الاجتماعية ومساعدتها للقطاعات الاكثر تعثرا مثل السيارات والبناء .  """]

# can it generalize?
test2 = ["""
وقال  خلال ازمة كورونا كانت غزة تصدر يوميا حمولة شاحنتين واكثر في بعض الاحيان من الملابس الواقية والكمامات لسوق الضفة والسوق الاسرائيلية , وعندما انخفض الطلب مؤخرا علي هذه الاصناف واصلت هذه المصانع مبيعاتها لهاتين السوقين من الاصناف الاخري , ومنها الملابس الرياضية وملابس الاطفال وحاليا يركز العديد من اصحاب مصانع الخياطة علي انتاج مستلزمات موسمي العيد والمدارس ومن بينهم كثيرون استانفوا اعمالهم من داخل منازلهم , حيث خصص البعض غرفة من منزله ليستخدمها في تشغيل عدد محدود من ماكينات الخياطة لتلبية بعض الطلبيات المتعلقة بصناعة اصناف محدده من الملابس  .	Fake
- عدد الاهداف التي سجلها كل فريق في الدوري هذا الموسم : 88 لريال و 88 لبرشلونة الذي خاض مباراة اقل من غريمه , علما بان برشلونة هو صاحب افضل هجوم حتي الان هذا الموسم في الدوري  """]

seq = token.texts_to_sequences(test1)
padded = pad_sequences(seq, maxlen=max_len)
pred = model.predict(padded)

pred
