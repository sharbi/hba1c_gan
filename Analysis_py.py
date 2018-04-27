
import pickle as pkl
import numpy as np
import seaborn as sns
import matplotlib

sns.set(context="poster", style='whitegrid', font='sans-serif')

X_real = pkl.load(open('./data/SPRINT/X_processed.pkl', 'rb'))
y_real = pkl.load(open('./data/SPRINT/y_processed.pkl', 'rb'))

print(X_real.shape, y_real.shape)

public_dir = './output/acgan_50_0.0002_100/'
# public_dir = './data/SPRINT/nonprivate/'

# private directory for multiple epoch testing - p7
#private_dir = output_dir + 'p15_8.0_0.0001_500_0.002_100'
private_dir = './data/SPRINT/private/'

#next
import h5py
import matplotlib.pyplot as plt
import pandas as pd

#matplotlib inline
plt.figure(figsize=(15,10))

hist = pkl.load(open(public_dir + 'acgan-history.pkl', 'rb'))
losses = ['loss', 'generation_loss', 'auxiliary_loss']

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        hist[p][g] = pd.DataFrame(hist[p][g], columns=losses)

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        plt.plot(hist[p][g]['generation_loss'], label='{} ({})'.format(g, p))

# get the NE and show as an equilibrium point
plt.hlines(-np.log(0.5), 0, hist[p][g]['generation_loss'].shape[0], label='Nash Equilibrium')
plt.legend(fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.ylabel(r'$L_s$',  fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('figure1_analysis.png')
plt.clf()


plt.figure(figsize=(15,10))
hist = pkl.load(open(private_dir + 'acgan-history.pkl', 'rb'))
losses = ['loss', 'generation_loss', 'auxiliary_loss']

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        hist[p][g] = pd.DataFrame(hist[p][g], columns=losses)

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        plt.plot(hist[p][g]['generation_loss'], label='{} ({})'.format(g, p))

# get the NE and show as an equilibrium point
plt.hlines(-np.log(0.5), 0, hist[p][g]['generation_loss'].shape[0], label='Nash Equilibrium')
plt.legend(fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.ylabel(r'$L_s$', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('figure2.png')
plt.clf()

#next

public_scores = pkl.load(open(public_dir + 'train_epoch_scores.p', 'rb'))
top_10_public = np.argsort(public_scores['rf'])[-5:]
top_10_lr = np.argsort(public_scores['lr'])[-5:]
top_10_public = np.concatenate([top_10_public, top_10_lr])
print(top_10_public)

#next_
np.random.seed(1)
# noisy selection
epoch_scores = pkl.load(open(private_dir + 'train_epoch_scores.p', 'rb'))

print(np.mean(np.argsort(epoch_scores['rf'])[-10:]))
print(np.mean(np.argsort(epoch_scores['lr'])[-10:]))

for j in np.argsort(epoch_scores['lr'])[-5:]:
    print(epoch_scores['lr'][j])

eps_per = 0.05
size = 500

top_5_rf = []
top_5_lr = []

for i in range(5):
    print(eps_per*size)
    noisy_epoch_scores_rf = (epoch_scores['rf'] +
                             (np.random.laplace(loc=0.0,
                                                scale=(1/(eps_per*size)),
                                                size=500)))

    j=1
    while(np.argsort(noisy_epoch_scores_rf)[-j:][0] in top_5_rf):
        j += 1
    top_5_rf.append(np.argsort(noisy_epoch_scores_rf)[-j:][0])
    size = size - 1

    noisy_epoch_scores_lr = (epoch_scores['lr'] +
                             (np.random.laplace(loc=0.0,
                                                scale=(1/(eps_per*size)),
                                                size=500)))
    k=1
    while(np.argsort(noisy_epoch_scores_lr)[-k:][0] in top_5_lr):
        k += 1
    top_5_lr.append(np.argsort(noisy_epoch_scores_lr)[-k:][0])
    size = size - 1


rf_noisy_scores = [epoch_scores['rf'][x] for x in top_5_rf]
lr_noisy_scores = [epoch_scores['lr'][x] for x in top_5_lr]
top_10 = np.concatenate([top_5_rf, top_5_lr])
print('top 10 ', top_10)
print(rf_noisy_scores, lr_noisy_scores)


rf_clean_scores = [epoch_scores['rf'][x] for x in np.argsort(epoch_scores['rf'])[-5:]]
lr_clean_scores = [epoch_scores['lr'][x] for x in np.argsort(epoch_scores['lr'])[-5:]]
print(rf_clean_scores, lr_clean_scores)
