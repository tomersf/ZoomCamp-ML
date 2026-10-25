#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[14]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[15]:


df = pd.read_csv('data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[16]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[17]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[18]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)


# In[19]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()


# In[19]:





# Accuracy and dummy model

# In[20]:


len(y_val)


# In[21]:


(y_val == churn_decision).sum()


# In[22]:


thresholds = np.linspace(0, 1, 21)
thresholds


# In[23]:


scores = []
for t in thresholds:
    churn_decision = (y_pred >= t)
    score = (y_val == churn_decision).mean()
    print("%.2f %.3f" % (t, score))
    scores.append(score)


# In[24]:


plt.plot(thresholds, scores, color='yellow')


# In[25]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[26]:


for t in thresholds:
    churn_decision = (y_pred >= t)
    score = accuracy_score(y_val, churn_decision)
    print("%.2f %.3f" % (t, score))


# In[27]:


from collections import Counter


# In[28]:


Counter(y_pred >= 1.0)


# In[29]:


1 - y_val.mean()


# Confusion table

# In[30]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[31]:


t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[32]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

(tp , tn)


# In[33]:


fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

(fp, fn)


# In[34]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

confusion_matrix


# In[35]:


(confusion_matrix / confusion_matrix.sum()).round(2)


# Precision and Recall

# In[36]:


# Precision - how many positive predictions were correct
p = tp / (tp + fp)
p


# In[37]:


# Recall - how many correct positive predictions out of total actual positive
r = tp / (tp + fn)
r


# In[37]:





# ROC Curves

# In[38]:


tpr = tp / (tp + fn) # True positive rate
tpr


# In[39]:


fpr = fp / (fp + tn) # False positive rate
fpr


# 

# In[40]:


thresholds = np.linspace(0, 1, 101)
thresholds


# In[41]:


scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, fp, fn, tn))


# In[42]:


columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)


# In[43]:


df_scores[::10] # every 10th record


# In[44]:


df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

df_scores[::10] # every 10th record


# In[45]:


plt.plot(df_scores.threshold, df_scores.tpr, color='blue', label='TPR')
plt.plot(df_scores.threshold, df_scores.fpr, color='orange', label='FPR')
plt.legend()


# In[45]:





# Random model

# In[46]:


np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))


# In[47]:


((y_rand > 0.5) == y_val).mean()


# In[48]:


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()
        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))
    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)
    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores


# In[49]:


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# In[50]:


df_rand[::10]


# In[51]:


plt.plot(df_rand.threshold, df_rand.tpr, color='blue', label='TPR')
plt.plot(df_rand.threshold, df_rand.fpr, color='orange', label='FPR')
plt.legend()


# In[51]:





# Ideal model

# In[52]:


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos


# In[53]:


y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal


# In[54]:


y_ideal_pred = np.linspace(0, 1, len(y_val))


# In[55]:


1 - y_val.mean()


# In[56]:


((y_ideal_pred >= 0.726) == y_ideal).mean()


# In[57]:


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)


# In[58]:


df_ideal[::10]


# In[59]:


plt.plot(df_ideal.threshold, df_ideal.tpr, color='blue', label='TPR')
plt.plot(df_ideal.threshold, df_ideal.fpr, color='orange', label='FPR')
plt.legend()


# In[59]:





# Putting all together

# In[60]:


plt.figure(figsize=(5,5))

plt.plot(df_scores.fpr, df_scores.tpr, color='blue', label='model')
plt.plot([0,1], [0,1], color='orange', label='random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


# In[61]:


from sklearn.metrics import roc_curve


# In[62]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[63]:


plt.figure(figsize=(5,5))

plt.plot(fpr, tpr, color='blue', label='model')
plt.plot([0,1], [0,1], color='orange', label='random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


# In[63]:





# ROV AUC

# In[64]:


from sklearn.metrics import auc

# AUC tells us the probability that randomly selected positive label has
# higher score than randomly selected negative label


# In[65]:


auc(fpr, tpr)


# In[66]:


auc(df_scores.fpr, df_scores.tpr)


# In[67]:


from sklearn.metrics import roc_auc_score


# In[68]:


roc_auc_score(y_val, y_pred)


# In[69]:


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


# In[70]:


import random


# In[71]:


n = 10000
success = 0

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success += 1

success / n


# In[72]:


# numpy style calculation
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)


# In[73]:


(pos[pos_ind] > neg[neg_ind]).mean()


# In[73]:





# Cross Validation
#  - Evaluating the samne model on different subsets of data
#  - Getting the average prediction and the spread within predictions

# In[74]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return dv, model


# In[75]:


dv, model = train(df_train, y_train)


# In[76]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[77]:


y_pred = predict(df_val, dv, model)


# In[78]:


from sklearn.model_selection import KFold


# In[79]:


kfold = KFold(n_splits=10, shuffle=True, random_state=1)


# In[80]:


scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)


# In[81]:


np.mean(scores), np.std(scores)


# In[81]:





# In[82]:


dv, model = train(df_full_train, df_full_train.churn.values)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# In[82]:





# Save the model

# In[83]:


import pickle

C = 1.0


# In[84]:


output_file = f'model_C={C}.bin'
output_file


# In[85]:


f_out = open(output_file, 'wb')


# In[86]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# In[86]:





# Load the model

# In[87]:


import pickle


# In[88]:


model_file = 'model_C=1.0.bin'


# In[89]:


with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


# In[91]:


dv, model


# In[105]:


customer = df_full_train.iloc[-5].to_dict()


# In[106]:


X = dv.transform([customer])


# In[107]:


model.predict_proba(X)[0, 1]


# In[104]:





# In[ ]:




