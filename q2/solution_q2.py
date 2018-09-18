from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import itertools
import networkx as nx
#scp -r -P 888 xyx@202.120.40.28:~/Downloads/shumo/input/raw/Q2_1.csv ./


# load data
terror=pd.read_csv('../input/raw/Q2_1.csv',encoding='ISO-8859-1')
print(terror[:4])
# 提取第二问的所有特征列，并转换成 list
#data_q2 = terror[['eventid', 'iyear']].values.tolist()
data_q2 = terror.values.tolist()

gname = terror['gname_10encode'].as_matrix()
print(gname[:10])
print(len(data_q2))
print(len(data_q2[0]))
print(data_q2[:2])


# seperate train and test
# test: 2015-2016, gname = 'unknown'
# train: 除了test 集的都是。考虑是否拿一点出来做validation。
# cleaned_dataset 第一列是event_id, 倒数第一列是goup_id。

cleaned_dataset = terror.as_matrix()
cleaned_dataset = terror.values.tolist()
print(type(cleaned_dataset))
row = len(cleaned_dataset)
cl = len(cleaned_dataset[0])
print(cleaned_dataset[:2])


# 新建一列 test ： 1表示属于test集， 0 表示属于train
append_ar = np.zeros((row,1), dtype='int64')
for idx, r in enumerate(cleaned_dataset):
    # 先拿出2015-2016
    # 再拿出unknown
    if (r[2] == 2015) or (r[2] == 2016):
        #gname = unkown =>>'0000000001'
        if r[-1] == 1.0:
            append_ar[idx] = 1 # 是测试样本
        else:
            append_ar[idx] = 0 # 2015-2016 间的已知gname，是训练样本
    elif r[-1] == 1.0:
        append_ar[idx] = 11 # 其他年份的unknown类型， 可能是训练样本，也可舍弃。
    else:
        append_ar[idx] = 12 # 其他年份的已知gname ,是训练样本。
# 在最后一列添加test标签
append_dataset = np.append(cleaned_dataset, append_ar,axis =1)
#print('add test tag')
#print(append_dataset[:2])
       
# 提取测试样本
test = np.array( [r[:-1] for r in append_dataset if r[-1] == 1 ])
num_test = len(test)
print(test[-10])

# 提取训练样本
train = np.array( [r[:-1] for r in append_dataset if r[-1] != 1])
num_train = len(train)
#print(train[:2])
print('feature dim %s' % len(train[0]))
print('train num is %s'% num_train)
print('test num is %s' % num_test)
print('total sum is %s '% row)
print(num_test + num_train)
# 提取不包含 ‘unknow’ 类型的训练样本
train_no_unknown = np.array( [r[:-1] for r in append_dataset if (r[-1] == 0 or r[-1] == 12)])
num_train_no_unknown = len(train_no_unknown)


# to generate groups
# row: group_id, line: event_id
def gen_groups(dataset):
    # dataset is a feature matrix , for each row, row[1] is eventid, row[0] is gname
    # return sorted gname, row[0] is gname, row[1] is eventid
    # return group_ids 
    print(dataset[:10])
    groups = dataset[dataset[:,0].argsort()]
    ngroup = []
    group_ids = []
    group_1 = [row[1] for row in groups if row[0] == 1]
    group_1.insert(0,1)
    print(group_1)
    if int(len(group_1)) != 1:
        ngroup.append(group_1)
    #print(group_1)
    group_id = 2
    temp = [group_id]
    tag = int(len(group_1)) - 1
    print(tag)
    for row in groups[tag:]:
        #print(row[0])
        if row[0] == group_id:
            temp.append(row[1])
            #print(temp)
        elif len(temp) != 1:
            #print(temp)
            ngroup.append(temp)
            group_id = group_id + 1
            temp = [group_id, row[1]]
        else:
            #print('temp empty when group_id = %d' % group_id)
            group_id = group_id + 1
            temp = [group_id]
    print('final goup_id is %d' % group_id)
    
    return ngroup

train_groups = gen_groups(train_no_unknown[:,[-1,0]])
print(train_groups[0])
    
    
# generate train dataset with pos_pairs, neg_pairs and labels
# asume we have got group_id and member list, we have a 2d list Groups, index is the groupid, row is all the eventid.

#for each group, generate negtive samples number
neg_count = 100

def gen_pairs(Groups, idx, neg_count):
    # to generate positive and negtive samples for the idx group
    # neg_percent: the number of negtive samples to generate = total negetive samples * neg_percent
    pos_pairs = [[x, y ] for idx_x, x in enumerate(Groups[idx]) for idx_y, y in enumerate(Groups[idx]) if idx_x != idx_y]
    num_pos = len(pos_pairs)
    pos_labels = np.ones(num_pos)
    
    # randomly generate negtive samples
    G = Groups
    #del G[idx]
    np.delete(G, (idx), axis=0)
    neg_pairs = []
    flat_G = [event for g in G for event in g]
    #neg_count = len(flat_G) * neg_percent
    for event in Groups[idx]:
        neg_idx = np.random.shuffle(list(range(0,neg_count)))
        print(neg_idx[:10])
        neg_data = G[neg_idx[:neg_count]]
        neg_pairs = neg_pairs + [[event, neg_event] for neg_event in neg_data]
    neg_labels = np.ones(len(neg_pairs))
    
    pairs = pos_pairs + neg_pairs 
    labels = pos_labels + neg_labels
    return pairs, labels
    
def gen_all_pairs(Groups):
    data = [gen_pairs(Groups, idx, neg_count) for idx,g in enumerate(Groups)]
    Data_pairs, Labels = zip(*data)
    return Data_pairs, Labels

def gen_train(Groups):
    # to gen test_groups and train_groups
    #
    #
    train_pairs, y_train0 = gen_all_pairs(train_groups)
    # concate the feature vector of two samples as one convated feature vector
    X_train1 = [np.append(pair[0], pair[1],axis =1) for pair in train_pairs] 
    print('number of x_train1 is %s' % len(X_train1))
    # 正反调换来一遍
    X_train2 = [np.append(pair[1], pair[0],axis =1) for pair in train_pairs] 
    # 两个特征差分试一下
    X_train = X_train1 + X_train2
    y_train = y_train0 + y_train0
    
    return X_train, y_train

gen_train(train_groups)    


# generate test dataset
def gen_test_pairs(fdata):
    # fdata is the feature matrix of test data
    # return concated feature, event_id pairs
    X_test = []
    test_ids = []
    for idx, event in enumerate(fdata):
        test_id = [[event[0],e[0]]  for e in fdata[idx+1:]]
        test_ids = test_ids + test_id
        
        test_feature = [np.append(event, e, axis = 1) for e in fdata[idx+1:]]
        X_test = X_test + test_feature
    
    return X_test, test_ids
        
        
    # xgboost//classifier

X_train, X_test, y_train, y_test = gen_train_test(Groups)
print(X_train[:2])
print(y_train[:2])
print(type(X_train))

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
'''
predictions = [round(value) for value in y_pred]
print(y_pred[:100])
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''


# # grouping the y_pred
# import matplotlib.pyplot as plt
# %matplotlib inline

# G = nx.read_edgelist('test_g.edgelist', nodetype=int, create_using=nx.DiGraph())
# G = G.to_undirected()
# nx.draw(G)

# g = nx.karate_club_graph()
# fig, ax = plt.subplots(1, 1, figsize=(8, 6));
# nx.draw_networkx(g, ax=ax)
