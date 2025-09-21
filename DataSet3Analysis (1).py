
# coding: utf-8

# In[20]:


import sklearn.model_selection
import sklearn.tree
import pandas
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import xgboost


# In[2]:


path = 'C:\\Users\\Alexey.Osipov\\Downloads\\'
tabHot = pandas.read_csv(path + 'HotStartData.csv', sep = ';')
tabCold = pandas.read_csv(path + 'ColdStartData.csv', sep = ';')


# In[3]:


tabHot = pandas.concat([tabHot.drop('GenderInfo', axis=1), pandas.get_dummies(tabHot['GenderInfo'])], axis=1)
tabCold = pandas.concat([tabCold.drop('GenderInfo', axis=1), pandas.get_dummies(tabCold['GenderInfo'])], axis=1)


# In[4]:


yHot = tabHot.Class
yCold = tabCold.Class
XHot = tabHot.drop('Class', axis=1)
XCold = tabCold.drop('Class', axis=1)


# In[5]:


X_train_Hot, X_test_Hot, y_train_Hot, y_test_Hot = sklearn.model_selection.train_test_split(XHot, yHot, random_state=239)
X_train_Cold, X_test_Cold, y_train_Cold, y_test_Cold = sklearn.model_selection.train_test_split(XCold, yCold, random_state=239)


# In[6]:


X_train_Hot.head(), y_train_Hot.head()


# In[24]:


#sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeClassifier(random_state=239), param_grid=[]
#                                     scoring=sklearn.metrics.average_precision_score, cv = 3)


# In[7]:


model = sklearn.tree.DecisionTreeClassifier(random_state=239)
model.fit(X_train_Hot, y_train_Hot)


# In[46]:


def modelFit(model, X_train, y_train):
    random_states = [128, 239, 28]
    stat_prec = 0
    stat_brier = 0
    stat_acc = 0
    for random_state in random_states:
        X_1, X_2, y_1, y_2 = sklearn.model_selection.train_test_split(X_train, y_train, random_state = random_state)
        model.fit(X_1, y_1)
        stat_prec += sklearn.metrics.average_precision_score(model.predict(X_2), y_2)
        stat_brier += sklearn.metrics.brier_score_loss(pandas.DataFrame(model.predict_proba(X_2))[1], y_2)
        stat_acc += sklearn.metrics.accuracy_score(model.predict(X_2), y_2)
    return stat_prec/3, stat_acc/3, stat_brier/3


# In[47]:


def modelCheck(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    values = model.predict(X_test)
    probs = pandas.DataFrame(model.predict_proba(X_test))
    vals = model.predict(X_test)
    return [sklearn.metrics.average_precision_score(y_test, vals),
    sklearn.metrics.accuracy_score(y_test, vals),
    sklearn.metrics.brier_score_loss(y_test, probs[1])]


# In[105]:


print(y_train_Hot.count())


# In[83]:


#estimate baseline
baselineTrain = []
print(y_train_Hot.count())
for i in range(0, y_train_Hot.count()):
    baselineTrain.append(0)
baselineTest = []
for i in range(0, y_test_Hot.count()):
    baselineTest.append(0)
baselineTrainprobs = []
print(y_train_Hot.count())
for i in range(0, y_train_Hot.count()):
    baselineTrainprobs.append(0.3)
baselineTestprobs = []
for i in range(0, y_test_Hot.count()):
    baselineTestprobs.append(0.3)

print([sklearn.metrics.average_precision_score(y_train_Hot, baselineTrain),
    sklearn.metrics.accuracy_score(y_train_Hot, baselineTrain),
    sklearn.metrics.brier_score_loss(y_train_Hot, baselineTrainprobs)])
print([sklearn.metrics.average_precision_score(y_test_Hot, baselineTest),
    sklearn.metrics.accuracy_score(y_test_Hot, baselineTest),
    sklearn.metrics.brier_score_loss(y_test_Hot, baselineTestprobs)])


# In[48]:


#trees only
modelTree = sklearn.tree.DecisionTreeClassifier(random_state=239)
print(modelFit(modelTree, X_train_Hot, y_train_Hot))
print(modelCheck(modelTree, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot))


# In[33]:


model = sklearn.tree.DecisionTreeClassifier(max_depth=7, random_state=239)
modelFit(model, X_train_Hot, y_train_Hot)


# In[34]:


modelCheck(model, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot)


# In[140]:


#RF only
modelRF = sklearn.ensemble.RandomForestClassifier(n_estimators=11,random_state=239, max_depth=20,min_samples_leaf=4, min_samples_split=0.1)
print(modelFit(modelRF, X_train_Hot, y_train_Hot))
print(modelCheck(modelRF, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot))


# In[141]:


modelRFCold = sklearn.ensemble.RandomForestClassifier(n_estimators=11,random_state=239, max_depth=20,min_samples_leaf=4, min_samples_split=0.1)
print(modelFit(modelRF, X_train_Cold, y_train_Cold))
print(modelCheck(modelRF, X_train_Cold, y_train_Cold, X_test_Cold, y_test_Cold))


# In[144]:


(modelRF.feature_importances_), X_train_Hot.columns


# In[106]:


from sklearn.tree import export_graphviz


# In[108]:


export_graphviz((modelRF.estimators_[0]), out_file = 'tree0.dot', rounded = True, precision = 1)


# In[103]:


modelRF.decision_path


# In[36]:


model2 = sklearn.ensemble.RandomForestClassifier(max_depth=3, n_estimators=100, random_state=239)
modelFit(model2, X_train_Hot, y_train_Hot)


# In[37]:


modelCheck(model2, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot)


# In[50]:


model3 = sklearn.linear_model.LogisticRegression(random_state=239)
print(modelFit(model3, X_train_Hot, y_train_Hot))
print(modelCheck(model3, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot))


# In[16]:





# In[51]:


model4 = sklearn.svm.SVC(probability=True, random_state=239)
print(modelFit(model4, X_train_Hot, y_train_Hot))
print(modelCheck(model4, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot))


# In[19]:





# In[165]:


model.feature_importances_, X_train_Hot.columns


# In[159]:


model.tree_.children_left


# In[52]:


modelnew = xgboost.XGBClassifier(max_depth=2, random_state=239, learning_rate=0.01, reg_lambda=5)
print(modelFit(modelnew, X_train_Hot, y_train_Hot))
print(modelCheck(modelnew, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot))


# In[26]:


modelnew = xgboost.XGBClassifier(max_depth=2, random_state=239, learning_rate=0.01, reg_lambda=5)
modelCheck(modelnew, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot)


# In[131]:


X_train_Hot.head()


# In[143]:


#we try to delete:
#(array([0.23820197, 0.14917393, 0.02341734, 0.17343024, 0.01618086,
#        0.05331723, 0.04089416, 0.19852072, 0.03420262, 0.01826004,
#        0.02292333, 0.02485296, 0.00462944, 0.00199515]),
# Index(['Duration', 'CreditHistory', 'Employment', 'Guarantors', 'Age',
#        'NumberOfCreditsInBank', 'JobInfo', 'NumberOfPeopleBeingLiable', 'Val',
#        'Telephone', 'Foreign', 'A91', 'A92', 'A93', 'A94'],
#       dtype='object'))
#gender info -> delete
#Foreign -> delete
#Employment -> delete


# In[159]:


X_train_Hot=X_train_Hot.drop('A91', axis=1)
X_train_Hot=X_train_Hot.drop('A92',  axis=1)
X_train_Hot=X_train_Hot.drop('A93',  axis=1)
X_train_Hot=X_train_Hot.drop('A94',  axis=1)
X_train_Hot=X_train_Hot.drop('Foreign',  axis=1)
X_test_Hot=X_test_Hot.drop('A91',  axis=1)
X_test_Hot=X_test_Hot.drop('A92',  axis=1)
X_test_Hot=X_test_Hot.drop('A93',  axis=1)
X_test_Hot=X_test_Hot.drop('A94',  axis=1)
X_test_Hot=X_test_Hot.drop('Foreign',  axis=1)


# In[160]:


X_test_Hot=X_test_Hot.drop('JobInfo',  axis=1)
X_train_Hot=X_train_Hot.drop('JobInfo',  axis=1)


# In[161]:


X_train_Hot.columns


# In[266]:


#RF only
modelRF = sklearn.ensemble.RandomForestClassifier(n_estimators=11,random_state=239, max_depth=20,min_samples_leaf=4, min_samples_split=0.1)
print(modelFit(modelRF, X_train_Hot, y_train_Hot))
print(modelCheck(modelRF, X_train_Hot, y_train_Hot, X_test_Hot, y_test_Hot))


# In[163]:


modelRF.feature_importances_, X_train_Hot.columns


# In[153]:


#cold case:
X_train_Cold=X_train_Cold.drop('A91', axis=1)
X_train_Cold=X_train_Cold.drop('A92',  axis=1)
X_train_Cold=X_train_Cold.drop('A93',  axis=1)
X_train_Cold=X_train_Cold.drop('A94',  axis=1)
X_train_Cold=X_train_Cold.drop('Foreign',  axis=1)
X_test_Cold=X_test_Cold.drop('A91',  axis=1)
X_test_Cold=X_test_Cold.drop('A92',  axis=1)
X_test_Cold=X_test_Cold.drop('A93',  axis=1)
X_test_Cold=X_test_Cold.drop('A94',  axis=1)
X_test_Cold=X_test_Cold.drop('Foreign',  axis=1)


# In[164]:


X_test_Cold=X_test_Cold.drop('JobInfo',  axis=1)
X_train_Cold=X_train_Cold.drop('JobInfo',  axis=1)


# In[267]:


#RF only
modelRF2 = sklearn.ensemble.RandomForestClassifier(n_estimators=11,random_state=239, max_depth=5,min_samples_leaf=5, min_samples_split=0.1)
print(modelFit(modelRF2, X_train_Cold, y_train_Cold))
print(modelCheck(modelRF2, X_train_Cold, y_train_Cold, X_test_Cold, y_test_Cold))


# In[166]:


modelRF2.feature_importances_, X_train_Cold.columns

