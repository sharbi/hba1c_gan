import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

X = pkl.load(open('./data/SPRINT/X_processed.pkl', 'rb'))
y = pkl.load(open('./data/SPRINT/y_processed.pkl', 'rb'))

print(X.shape, y.shape)

#next_
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

training_size = 300

X_train = X[:training_size]
y_train = y[:training_size]

X_test = X[training_size:]
y_test = y[training_size:]

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

classifier = RandomForestClassifier()
y_pred = classifier.fit(X_train, y_train).predict(X_test)

accuracy_score(y_test, y_pred)

#next_
noise = np.random.normal(0, 1, 600)
plt.hist(noise, bins=30)
print(np.where(abs(noise)<0.01))

#next_
directory = './output/acgan_50_0.0002_100/'
#directory = './output/4.0_0.0001_500_0.002_100/'
#directory = './data/SPRINT/nonprivate/'

#next_
import h5py
import matplotlib.pyplot as plt
import pandas as pd

#matplotlib inline
plt.figure(figsize=(10,10))

hist = pkl.load(open(directory + 'acgan-history.pkl', 'rb'))
losses = ['loss', 'generation_loss', 'auxiliary_loss']

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        hist[p][g] = pd.DataFrame(hist[p][g], columns=losses)

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        plt.plot(hist[p][g]['generation_loss'], label='{} ({})'.format(g, p))

# get the NE and show as an equilibrium point
plt.hlines(-np.log(0.5), 0, hist[p][g]['generation_loss'].shape[0], label='Nash Equilibrium')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel(r'$L_s$')

plt.savefig('figure1_predictGroup.png')
plt.clf()

# for p in hist['privacy']:
#     print(p)

#next_
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

acgan = __import__('ac_gan')

from glob import glob
from keras.models import load_model
from sklearn import linear_model

latent_size = 100


lr_clf = linear_model.LogisticRegression()
transfer_clf = RandomForestClassifier()

mean_scores = []
lr_scores = []
#directory = './output/p' + str(9) + '_8.0_0.0001_500_0.002_100/'
# directory = './output/acgan_500_0.0002_100/'

for i in range(0, 4):
    gen_name = sorted(glob(directory + 'params_generator*'))[i]
    print(gen_name)
    g = load_model(gen_name)

    generate_count = training_size

    noise = np.random.uniform(-1, 1, (generate_count, latent_size))
    sampled_labels = np.random.randint(0, 2, generate_count)
    generated_images = g.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)

    gen_X_train = np.reshape(generated_images, (training_size, 3, 12))
    gen_X_train = gen_X_train.astype(int)
    gen_X_train = gen_X_train.clip(min=0)

    gen_X_train = gen_X_train.reshape(generate_count, -1)
    gen_y_train = sampled_labels


    mean_scores.append(accuracy_score(y_train, transfer_clf.fit(gen_X_train, gen_y_train).predict(X_train)))
    lr_scores.append(accuracy_score(y_train, lr_clf.fit(gen_X_train, gen_y_train).predict(X_train)))
pkl.dump({'rf': mean_scores, 'lr': lr_scores}, open(directory + 'train_epoch_scores.p', 'wb'))
