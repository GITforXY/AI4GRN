from sklearn.metrics import roc_auc_score

def evaluate_auc(test_pred, test_true):

    return roc_auc_score(test_true, test_pred)
