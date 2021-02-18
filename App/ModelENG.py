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

real = pd.read_csv('/datasets/en/True.csv')
fake = pd.read_csv('/datasets/en/Fake.csv')

real.shape
real.head

fake_drop = fake.drop(fake.loc[fake.text == ' '].index)
real_drop = real.drop(real.loc[real.text == ' '].index)

# Give labels to data before combining
fake_drop['fake'] = 1
real['fake'] = 0

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
test1 = [""" Reduced to their weakest state in a generation, Democratic Party leaders will gather in two cities this weekend to plot strategy and select a new national chairman with the daunting task of rebuilding the partyâ€™s depleted organization. But senior Democratic officials concede that the blueprint has already been chosen for them  â€”   by an incensed army of liberals demanding no less than total war against President Trump. Immediately after the November election, Democrats were divided over how to handle Mr. Trump, with one camp favoring   confrontation and another backing a seemingly less risky approach of coaxing him to the center with offers of compromise. Now, spurred by explosive protests and a torrent of angry phone calls and emails from constituents  â€”   and outraged themselves by Mr. Trumpâ€™s swift moves to enact a   agenda  â€”   Democrats have all but cast aside any notion of conciliation with the White House. Instead, they are mimicking the Republican approach of the last eight years  â€”   the â€œparty of noâ€  â€”   and wagering that brash obstruction will pay similar dividends. Gov. Jay Inslee of Washington, vice chairman of the Democratic Governors Association, said there had been a â€œtornado of supportâ€ for    resistance to Mr. Trump. Mr. Inslee, who backed a lawsuit against the presidentâ€™s executive order banning refugee admissions and travel from seven   countries, said Democrats intended to send a stern message to Mr. Trump during a conference of governors in the nationâ€™s capital. â€œMy belief is, we have to resist every way and everywhere, every time we can,â€ when Mr. Trump offends core American values, Mr. Inslee said. By undermining Mr. Trump across the board, he said, Democrats hope to split Republicans away from a president of their own party. â€œUltimately, weâ€™d like to have a few Republicans stand up to rein him in,â€ Mr. Inslee said. â€œThe more air goes out of his balloon, the earlier and likelier that is to happen. â€ Yet Democrats acknowledge there is a wide gulf between the partyâ€™s desire to fight Mr. Trump and its power to thwart him, quietly worrying that the expectations of the partyâ€™s activist base may outpace what Democratic lawmakers can achieve. â€œThey want us to impeach him immediately,â€ said Representative John Yarmuth, Democrat of Kentucky. â€œAnd of course we canâ€™t do that by ourselves. â€ Some in the party also fret that a posture of unremitting hostility to the president could imperil lawmakers in red states that Mr. Trump won last year, or compromise efforts for Democrats to present themselves to moderate voters as an inoffensive alternative to the polarizing president. Rarely have Democrats been so weakened. Republicans control the White House, both chambers of Congress and 33 governorships, and they are preparing to install a fifth conservative, Neil M. Gorsuch, on the Supreme Court. Further, because of changes to Senate rules that were enacted under Democratic control, the party has been unable to block Mr. Trumpâ€™s cabinet nominees from being confirmed by a simple majority vote. Democrats, in other words, have few instruments at the moment to wound Mr. Trumpâ€™s administration in the manner their core voters are demanding. Still, a mood of stiff opposition has taken hold on Capitol Hill, with Democrats besieged by constituents enraged by Mr. Trumpâ€™s actions  â€”   and lawmakers sharing their alarm. â€œWe have to fight like hell to stop him and hopefully save our country,â€ said Senator Jeff Merkley, Democrat of Oregon, echoing the   stakes liberal voters are giving voice to at crowded town hall meetings. Senator Thomas R. Carper of Delaware, a     Democrat up for   in 2018, cautioned that loathing Mr. Trump, on its own, was not a governing strategy. He said he still hoped for compromise with Republicans on infrastructure funding and perhaps on a plan to improve or â€œrepairâ€ the Affordable Care Act. â€œThere is this vitriol and dislike for our new president,â€ Mr. Carper said. â€œThe challenge for us is to harness it in a productive way and a constructive way, and I think we will. â€ But Mr. Carper said the deliberations over Mr. Trumpâ€™s cabinet appointments had woken up Democrats, recalling that he had heard from thousands of voters about Scott Pruitt, Mr. Trumpâ€™s Environmental Protection Agency administrator, and Betsy DeVos, his education secretary. Virtually every message expressed seething opposition, he said. At times, Democratic frustration with Mr. Trump has already flared well beyond the normal range of opposition discourse: In Virginia, Tom Perriello, a former congressman seeking his partyâ€™s nomination for governor, apologized after calling Mr. Trumpâ€™s election a â€œpolitical and constitutional Sept. 11. â€ And in New Jersey, Phil Murphy, a former Goldman Sachs banker and ambassador to Germany, drew criticism in his campaign for governor after likening the current political moment in America to the rise of Adolf Hitler. Among    Democrats, however, it is far from clear that the rhetoric of heated opposition is unwelcome. A survey published on Wednesday by the Pew Research Center found that nearly   of Democrats said they were concerned the party would not do enough to oppose Mr. Trump only 20 percent were concerned Democrats would go too far in opposition. A handful of liberal groups have already sprung up threatening to wage primary challenges against incumbent Democrats whom they see as insufficiently militant against Mr. Trump, raising the prospect of the same internecine wars that plagued Republicans during President Barack Obamaâ€™s administration. In the race for the chairmanship of the Democratic National Committee, which concludes with a vote in Atlanta on Saturday, the restive mood of liberal activists has buoyed a pair of insurgents, Representative Keith Ellison of Minnesota and Mayor Pete Buttigieg of South Bend, Ind. against the perceived   Thomas E. Perez. Mr. Perez, who was Mr. Obamaâ€™s labor secretary, is still viewed as a favorite in the race, and he has been backed by former Vice President Joseph R. Biden Jr. But he has struggled to dispel the impression that he is an anointed favorite of Washington power brokers. And Mr. Ellison and Mr. Buttigieg have continued to collect   endorsements: Mr. Ellison won the support of Representative John Lewis of Georgia, the civil rights leader, on Tuesday, and Mr. Buttigieg was endorsed Wednesday by Howard Dean, the former party chairman who remains admired on the left. In a sign of how little heed Democrats are paying to traditional forces, Mr. Ellison remains viable despite being bluntly attacked as â€œan  â€ by Haim Saban, one of the most prolific donors to the party and its candidates. Christine C. Quinn, a vice chairwoman of the New York State Democratic Committee, who was a prominent surrogate for Hillary Clinton last year, said she backed Mr. Ellison, who was the first Muslim elected to Congress, in part because of the forcefulness of his criticism of the White House. â€œThis is not a normal Republican president, and these are not normal times,â€ said Ms. Quinn, a former speaker of the New York City Council. â€œThis isnâ€™t a time for polite parties anymore. This is a time to take a different posture of true aggressiveness. â€ Martin Oâ€™Malley, a former Maryland governor who has endorsed Mr. Buttigieg, said impatient Democrats might challenge even members of their own party in their enthusiasm to take on Mr. Trump. Mr. Oâ€™Malley said the party base plainly wanted leaders who would be â€œwilling to fight the fight and where necessary filibuster and otherwise obstruct. â€ He said he expected younger,   liberals to run against some Democratic incumbents as well as Republicans. â€œThatâ€™s a good thing, and itâ€™s overdue,â€ he said. So far, the most prominent leaders of the Democratic Partyâ€™s activist wing, including Senators Elizabeth Warren of Massachusetts and Bernie Sanders of Vermont, have not encouraged challenges to sitting Democratic lawmakers who have accommodated Mr. Trump. Mr. Merkley, an ally of Mr. Sanders, suggested liberals seeking scalps would get no help from progressive senators if they try to unseat Democratic senators from conservative Missouri, Montana, North Dakota and West Virginia, calling those lawmakers â€œperfectly suited to those states. â€ Two mayors in Democratic cities, however, have gotten a taste of what awaits those who do not bow completely to the demands of the   forces: When Carolyn Goodman of Las Vegas, a Democrat turned independent, and Levar Stoney of Richmond, Va. a Democrat, resisted deeming their municipalities â€œsanctuary cities,â€ each was met with anger from supporters of expanding protection against deportation for undocumented immigrants. â€œThey want change to happen overnight,â€ Mr. Stoney said of the newly energized activists. Nowhere is it more clear, however, that the protesters are leading the politicians than on Capitol Hill. Senate Democratic leaders had hoped to capitalize on Mr. Trumpâ€™s nomination of Tom Price as health secretary by assailing Republicans for wanting to trim Medicare, an issue Democrats aim to run on in 2018. But Mr. Price was vastly overshadowed by the nomination of Ms. DeVos, who galvanized the new activists like no other cabinet pick. â€œPart of what I think the Bernie campaign taught us, even the Trump campaign taught us, and now the resistance is teaching us, is just ditch the consultants and consult with your conscience and constituents first,â€ said Senator Brian Schatz of Hawaii, warning his fellow Democrats that â€œitâ€™s a foolâ€™s errand to try to plan this out like itâ€™s a traditional political operation. â€ Mr. Merkley boasted that â€œweâ€™re doing things in the Senate that are less conventional,â€ efforts he said were aimed at conveying to   voters that â€œhey, weâ€™re here and weâ€™re fighting. â€ Those efforts have included tactics like walking out on nomination hearings and opposing even less controversial cabinet appointments, such as that of Transportation Secretary Elaine Chao, the wife of the Senate Republican leader, Mitch McConnell. The fear factor is real, said Adam Jentleson, a former Senate Democratic aide. Images of angry constituents jeering Senator Sheldon Whitehouse, a reliable liberal from Rhode Island, at a town   meeting in late January for supporting the selection of Mike Pompeo as C. I. A. director quickly circulated among other Democratic senators, he said. â€œIt was â€ Mr. Jentleson said, â€œbecause it made clear that the base is not going to let them off the hook. â€ """]

seq = token.texts_to_sequences(test1)
padded = pad_sequences(seq, maxlen=max_len)
pred = model.predict(padded)

pred

model.save('/content/gdrive/My Drive/fake-news/my_model')
