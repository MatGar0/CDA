# Applies smote on given classifier and dataset
def smote_testing_for_classifier(x, y, classifier,filename):  
    from imblearn.over_sampling import SMOTE 
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    from testing_metrics import print_confusion
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)#test_size: proportion of train/test data
    #y_score_no_sampling = classifier.fit(x_train, y_train).decision_function(x_test)
    classifier.fit(x_train, y_train)
    y_pred_no_sampling = classifier.predict(x_test)
    print_confusion(y_test,y_pred_no_sampling, filename+' with no sampling')
    y_score_no_sampling = classifier.decision_function(x_test)
    
    sm = SMOTE(ratio=0.1)
    x_train10, y_train10 = sm.fit_sample(x_train, y_train)
    #y_score_10 = classifier.fit(x_train, y_train).decision_function(x_test)
    classifier.fit(x_train10, y_train10)
    y_pred_10 = classifier.predict(x_test)
    print_confusion(y_test,y_pred_10, filename+' with 10% sampling')    
    y_score_10 = classifier.decision_function(x_test)
        
    sm = SMOTE(ratio=0.2)
    x_train20, y_train20 = sm.fit_sample(x_train, y_train)
    classifier.fit(x_train20, y_train20)
    y_pred_20 = classifier.predict(x_test)
    print_confusion(y_test,y_pred_20, filename+' with 20% sampling')    
    y_score_20 = classifier.decision_function(x_test)
    
    sm = SMOTE(ratio=0.3)
    x_train30, y_train30 = sm.fit_sample(x_train30, y_train30)
    classifier.fit(x_train30, y_train30)
    y_pred_30 = classifier.predict(x_test)
    print_confusion(y_test,y_pred_30, filename+' with 30% sampling')    
    y_score_30 = classifier.decision_function(x_test)
    
    sm = SMOTE(ratio=0.4)
    x_train40, y_train40 = sm.fit_sample(x_train, y_train)
    #y_score_40 = classifier.fit(x_train, y_train).decision_function(x_test)
    classifier.fit(x_train40, y_train40)
    y_pred_40 = classifier.predict(x_test)
    print_confusion(y_test,y_pred_40, filename+' with 40% sampling')    
    y_score_40 = classifier.decision_function(x_test)

    sm = SMOTE(ratio=0.5)
    x_train50, y_train50 = sm.fit_sample(x_train, y_train)
    #y_score_50 = classifier.fit(x_train, y_train).decision_function(x_test)
    classifier.fit(x_train50, y_train50)
    y_pred_50 = classifier.predict(x_test)
    print_confusion(y_test,y_pred_50, filename+' with 50% sampling')   
    y_score_50 = classifier.decision_function(x_test)
        
    #%
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr[0], tpr[0], _ = metrics.roc_curve(y_test[:], y_score_no_sampling[:])
    roc_auc[0] = metrics.auc(fpr[0], tpr[0])
    fpr[0], tpr[0], _ = metrics.roc_curve(y_test[:], y_score_no_sampling[:])
    roc_auc[0] = metrics.auc(fpr[0], tpr[0])
    
    fpr[1], tpr[1], _ = metrics.roc_curve(y_test[:], y_score_10[:])
    roc_auc[1] = metrics.auc(fpr[1], tpr[1])
    fpr[1], tpr[1], _ = metrics.roc_curve(y_test[:], y_score_10[:])
    roc_auc[1] = metrics.auc(fpr[1], tpr[1])
    
    fpr[2], tpr[2], _ = metrics.roc_curve(y_test[:], y_score_20[:])
    roc_auc[2] = metrics.auc(fpr[2], tpr[2])
    fpr[2], tpr[2], _ = metrics.roc_curve(y_test[:], y_score_20[:])
    roc_auc[2] = metrics.auc(fpr[2], tpr[2])
    
    fpr[3], tpr[3], _ = metrics.roc_curve(y_test[:], y_score_30[:])
    roc_auc[3] = metrics.auc(fpr[3], tpr[3])
    fpr[3], tpr[3], _ = metrics.roc_curve(y_test[:], y_score_30[:])
    roc_auc[3] = metrics.auc(fpr[3], tpr[3])
    
    fpr[4], tpr[4], _ = metrics.roc_curve(y_test[:], y_score_40[:])
    roc_auc[4] = metrics.auc(fpr[4], tpr[4])
    fpr[4], tpr[4], _ = metrics.roc_curve(y_test[:], y_score_40[:])
    roc_auc[4] = metrics.auc(fpr[4], tpr[4])
    
    fpr[5], tpr[5], _ = metrics.roc_curve(y_test[:], y_score_50[:])
    roc_auc[5] = metrics.auc(fpr[5], tpr[5])
    fpr[5], tpr[5], _ = metrics.roc_curve(y_test[:], y_score_50[:])
    roc_auc[5] = metrics.auc(fpr[5], tpr[5])
    
    import matplotlib.pyplot as plt
    # Plot ROC curve
    plt.figure()
    
    plt.plot(fpr[0], tpr[0], label='ROC curve with no sampling (area = {1:0.2f})'
                                       ''.format(0, roc_auc[0]))
    plt.plot(fpr[1], tpr[1], label='ROC curve with sampling ratio of 10% (area = {1:0.2f})'
                                       ''.format(1, roc_auc[1]))
    plt.plot(fpr[2], tpr[2], label='ROC curve with sampling ratio of 20% (area = {1:0.2f})'
                                       ''.format(2, roc_auc[2]))
    plt.plot(fpr[3], tpr[3], label='ROC curve with sampling ratio of 30% (area = {1:0.2f})'
                                       ''.format(3, roc_auc[3]))
    plt.plot(fpr[4], tpr[4], label='ROC curve with sampling ratio of 40% (area = {1:0.2f})'
                                       ''.format(4, roc_auc[4]))
    plt.plot(fpr[5], tpr[5], label='ROC curve with sampling ratio of 50% (area = {1:0.2f})'
                                       ''.format(5, roc_auc[5]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for classifiers with different SMOTE sampling ratio')
    plt.legend(loc="lower right",prop={'size':6})
    plt.savefig(filename+'_ROC.png', format='png', dpi=1000)
    
    plt.show()
    