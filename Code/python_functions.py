from random import randrange
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
def CVgeneric(classifier,training_set,k,loss_function):
    x = KFold(n_splits=k)
    d = x.split(training_set.index)
    df = {}
    df['Accuracy'] = []
    df['Fold'] = []
    to_plot_data = []
    k = 1
    for train,test in d:
        train_data = training_set.iloc[train]
        test_data = training_set.iloc[test]
        mod = classifier.fit(X=train_data.drop(columns='expert_label'),y=train_data['expert_label'])
        pred = classifier.predict(test_data.drop(columns='expert_label'))
        truth = test_data['expert_label']
        df['Accuracy'].append(classifier.score(test_data.drop(columns='expert_label'),test_data['expert_label']))
        df['Fold'].append(k)
        k += 1
        #loss = loss_function(train_data)
    mod = classifier.fit(X=training_set.drop(columns='expert_label'),y=training_set['expert_label'])
    pred = classifier.predict(training_set.drop(columns='expert_label'))
    probs = classifier.predict_proba(training_set.drop(columns='expert_label'))
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(training_set['expert_label'], preds)
    roc_auc = metrics.auc(fpr, tpr)
    df['Accuracy'].append(np.mean(df['Accuracy']))
    df['Fold'].append('Average')

    return [pd.DataFrame.from_dict(df),[fpr,tpr,threshold,roc_auc]]
