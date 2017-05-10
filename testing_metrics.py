# Prints classification results
def print_confusion(y_test,y_predict,classifier_name):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(y_predict)):
        if y_test[i]==1 and y_predict[i]==1:
            TP += 1
        if y_test[i]==0 and y_predict[i]==1:
            FP += 1
        if y_test[i]==1 and y_predict[i]==0:
            FN += 1
        if y_test[i]==0 and y_predict[i]==0:
            TN += 1
    print ('The confusion matrix for ' +classifier_name )
    print ('TP: '+ str(TP))
    print ('FP: '+ str(FP))
    print ('FN: '+ str(FN))
    print ('TN: '+ str(TN))
    
def print_prec_recall(y_test,y_predict,classifier_name):
    from sklearn.metrics import precision_recall_curve
    print ('Precision recall curve for ' +classifier_name )
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
