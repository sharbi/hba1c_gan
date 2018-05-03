import pandas as pd
import numpy as np

baseline = pd.read_csv('./data/SPRINT/baseline.csv')
print(baseline.head())
print(baseline.shape)

bp = pd.read_csv('./data/SPRINT/bp.csv')
print(bp.head())
print(bp.shape)

print(bp['MASKID'].unique())
print(len(bp['VISITCODE'].unique()))
print(bp['VISITCODE'].unique())
print(baseline['INTENSIVE'].value_counts())


i = 0
y = {}
X = {}

visit_codes = ['RZ', '1M', '2M', '3M', '6M', '9M', '12M', '15M',
               '18M', '21M', '24M', '27M', '30M']
print(len(visit_codes))

for p in bp['MASKID'].unique():
    x_ex = np.zeros([3, 13], dtype='int')

    for j in range(len(visit_codes)):
        sbp = bp.loc[(bp['MASKID'] == p) & (bp['VISITCODE'] == visit_codes[j])]['SBP']
        dbp = bp.loc[(bp['MASKID'] == p) & (bp['VISITCODE'] == visit_codes[j])]['DBP']
        n_bpclasses = bp.loc[(bp['MASKID'] == p) & (bp['VISITCODE'] == visit_codes[j])]['N_BPCLASSES']

        if (len(sbp) == 1 and len(dbp) == 1 and len(n_bpclasses) == 1 and not sbp.isnull().values.any()
            and not dbp.isnull().values.any() and not n_bpclasses.isnull().values.any()):
            x_ex[0, j] = sbp
            x_ex[1, j] = dbp
            x_ex[2, j] = n_bpclasses

    y[p[1:]] = baseline.loc[baseline['MASKID'] == p]['INTENSIVE']
    X[p[1:]] = x_ex

    print(i)
    i += 1

# next set
print(bp['SBP'].max(), bp['DBP'].max(), bp['N_BPCLASSES'].max())

# next set

import pickle as pkl
pkl.dump(X, open('./data/SPRINT/X_bp.pkl', 'wb'))
pkl.dump(y, open('./data/SPRINT/y_bp.pkl', 'wb'))

#next
import numpy as np

i = 0

X_array = np.array([])
y_array = np.array([])

for key, value in sorted(X.items()):
    if 0 not in value[1, :12]:
        if X_array.shape[0] == 0:
            X_array = np.array([value[:, :12]])
            y_array = y[key]
        else:
            X_array = np.concatenate((X_array, [value[:, :12]]))
            y_array = np.concatenate((y_array, y[key]))

#next
pkl.dump(X_array, open('./data/SPRINT/X_processed.pkl', 'wb'))
pkl.dump(y_array, open('./data/SPRINT/y_processed.pkl', 'wb'))

#next

print(bp['SBP'].max(), bp['DBP'].max(), bp['N_BPCLASSES'].max())

sbp_max = bp['SBP'].max()
dbp_max = bp['DBP'].max()
n_bp_max = bp['N_BPCLASSES'].max()

import pickle as pkl
X = pkl.load(open('./data/SPRINT/X_bp.pkl', 'rb'))
y = pkl.load(open('./data/SPRINT/y_bp.pkl', 'rb'))

import numpy as np

i = 0

X_array = np.array([], dtype=float)
y_array = np.array([])

for key, value in sorted(X.items()):
    if 0 not in value[0, :12]:
        if X_array.shape[0] == 0:
            X_array = np.array([value[:, :12]], dtype=float)
            y_array = y[key]
        else:
            X_array = np.concatenate((X_array, [value[:, :12]]))
            y_array = np.concatenate((y_array, y[key]))

# Make normalization between -1 and 1
sbp_normalization_value = float(sbp_max) / 2
dpb_normalization_value = float(dbp_max) / 2
n_bp_normalization_value = float(n_bp_max) / 2

#next
for val in X_array:
    val[0] = val[0] - sbp_normalization_value / sbp_normalization_value
    val[1] = val[1] - dpb_normalization_value / dpb_normalization_value
    val[2] = val[2] - n_bp_normalization_value / n_bp_normalization_value 

print(X_array[:3])

#next
pkl.dump(X_array, open('./data/SPRINT/X_normalized.pkl', 'wb'))
pkl.dump(y_array, open('./data/SPRINT/y_normalized.pkl', 'wb'))
