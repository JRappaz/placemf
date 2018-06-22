import os,sys,time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable
from torch import nn,LongTensor,FloatTensor
import torch.utils.data as data
from tqdm import tqdm
from collections import Counter
from scipy.spatial.distance import cdist
import datetime
import torch.nn.functional as F

MIN_INTER             = 5
ROW_START             = 10000000
ROW_END               = 15000000
CUDA                  = False
BATCH_SIZE            = 512
SAMPLES_PER_ITERATION = 100000

class DataStore():
  def __init__(self,fn):

    self.num_test       = 10
    self.interactions = []
    self.max_u = 1
    self.max_i = 1
    self.user_inter     = {}
    self.tot_items      = set()
    self.tot_items_test = set()
    self.train          = {}
    self.train_inter    = []
    self.test           = {}
    self.valuemap       = {}
    self.load_items(fn)

  def load_items(self,fn):
    self.df = pd.read_csv(fn)
    self.df = self.df.iloc[ROW_START:ROW_END]

    print("Raw interactions: ",len(self.df))
    print("Raw user count: ", len(set(self.df['user'].tolist())) )

    filtered = self.df['user'].value_counts()
    filtered = filtered[filtered>=MIN_INTER]

    self.df = self.df[self.df['user'].isin(filtered.index)]
    print("Filtered interactions: ", len(self.df))
    print("Filtered user count: ",   len(set(self.df['user'].tolist())) )

    adj_labels = ['a'+str(i) for i in range(1,10)]
    self.max_u = int(self.df[['user']+adj_labels].max().max())

    self.test  = self.df.groupby('user', group_keys=False).apply(lambda df: df.sample(1))
    self.train = self.df.drop(self.test.index)

  def sample(self,u):

    while True:
      rand_row = self.train.sample(1)
      if int(rand_row['user']) != u:
        break

    return rand_row

  def batch_sample(self):
    batch = []

    testpos = self.train.sample(SAMPLES_PER_ITERATION)
    testneg = self.train.sample(SAMPLES_PER_ITERATION)

    for ind in range(0,len(testpos)):

      pos = testpos.iloc[ind].tolist()
      neg = testneg.iloc[ind].tolist()

      if pos[1]==neg[1]:
        neg = self.sample(pos[1]).iloc[0].tolist()

      batch.append(pos + neg)

    batch = FloatTensor(batch)

    if CUDA:
      batch = batch.cuda()

    return torch.utils.data.DataLoader(batch, batch_size=BATCH_SIZE, shuffle=True)

  def filter_pos_inter(self,user,recs):
    for i in self.train[user]:
      try:
        recs.remove(i)
      except:
        pass
    return recs

class PlaceModel(nn.Module):
  def __init__(self,datastore):
    super(BPR, self).__init__()
    self.logs      = False

    self.K         = 10
    self.user_embs = nn.Embedding(datastore.max_u+1, self.K)

    if CUDA:
      self.user_embs = self.user_embs.cuda()

    self.n_iter    = 1000
    self.optimizer = optim.SGD(self.parameters(), lr=0.1, weight_decay=0.01)
    self.datastore = datastore

  def forward(self,user,nearby):
    # Set interactions with no users to zero
    self.user_embs.weight.data[0,:] = 0
    preds  = (self.user_embs(user) * self.user_embs(nearby).sum(1)).sum(1)

    return preds

  def bpr_loss(self,pos_preds,neg_preds):
    sig = nn.Sigmoid()
    return (1.0 - sig(pos_preds - neg_preds)).pow(2).sum()

  def auc_test(self):

    testpos = self.datastore.test.sample(SAMPLES_PER_ITERATION)
    testneg = self.datastore.df.sample(SAMPLES_PER_ITERATION)

    batch = []
    for ind in tqdm(range(0,len(testpos))):

      pos = testpos.iloc[ind].tolist()
      neg = testneg.iloc[ind].tolist()

      if pos[1]==neg[1]:
        neg = self.datastore.sample(pos[1]).iloc[0].tolist()

      batch.append(pos + neg)

    batch = FloatTensor(batch)

    if CUDA:
      batch = batch.cuda()

    users     = Variable(batch[:,1]).long()
    pos_items = Variable(batch[:,5:14]).long()
    neg_items = Variable(batch[:,19:]).long()

    pos_preds = self(users,pos_items)
    neg_preds = self(users,neg_items)

    auc = 0.0
    for i in range(0,len(pos_preds)):
      sp = pos_preds[i]
      sn = neg_preds[i]

      if CUDA:
        sp = sp.data.cpu().numpy()[0]
        sn = sn.data.cpu().numpy()[0]

      if sp > sn:
        auc += 1.0
      elif sp==sn:
        auc += 0.5

    return auc / len(testpos)

  def train(self):

    for epoch in range(self.n_iter):
      print("epoch: ",epoch)
      running_loss = 0.0
      batch_sample = self.datastore.batch_sample()
      for data in tqdm(batch_sample):

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize

        users     = Variable(data[:,1]).long()
        pos_items = Variable(data[:,5:14]).long()
        neg_items = Variable(data[:,19:]).long()

        pos_preds = self(users,pos_items)
        neg_preds = self(users,neg_items)

        loss = self.bpr_loss(pos_preds,neg_preds)
        loss.backward()

        self.optimizer.step()

        # print statistics
        running_loss += loss.data[0]

      if epoch%10==0 and epoch>0:
        print('AUC test: ', self.auc_test())


ds = DataStore('data/tiles_adjacency.csv')

mod = PlaceModel(ds)
mod.train()

