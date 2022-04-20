import cnn
import keras
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from allfiles import trainall,testall,trlabelall,telabelall

roc=[]
pr=[]
acc=[]
sen=[]
spe=[]
Mcc=[]
test_loss=[]
test_acc=[]
predict_test=[]
testlabel_test=[]

for i in range(len(trainall)):      ##0-9的数   十份中的第i+1份
    print(i)
    model = keras.models.load_model('cnn.model1')
    train = np.expand_dims(trainall[i], 2)   ##训练集特征
    test = np.expand_dims(testall[i],2)      ##测试集特征
    trainlabel = trlabelall[i]     ##训练集的标签
    testlabel = telabelall[i]      ##测试集的标签
    model.fit(train, trainlabel,
              batch_size=512,  # 每个batch的大小为512
              epochs=80,  # 在全数据集上迭代20次
              )

    eval = model.evaluate(test, testlabel, verbose=0)
    print("Evaluation on test data: loss = %0.5f accuracy = %0.3f%% \n" \
          % (eval[0], eval[1] * 100))

    np.set_printoptions(precision=5)         ##控制输出的小数点个数是5
    predict_y2 = model.predict_proba(test)   ##预测概率  y_scores
    index_yy = np.where(predict_y2 < 0.5, 0, predict_y2)
    index_y2 =np.where(index_yy >= 0.5, 1, index_yy)    ##预测标签

    for m in predict_y2:
        predict_test.append(m)
    for n in testlabel:
        testlabel_test.append(n)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(testlabel)):
        if testlabel[j] ==1:
            if testlabel[j] == index_y2[j]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if testlabel[j] == index_y2[j]:
                tn = tn +1
            else:
                fp = fp + 1
    accuracy = float(tp + tn)/len(testlabel)
    if tp == 0 and fp == 0:
         MCC = 0
         sensitivity = float(tp)/ (tp+fn)
         specificity = float(tn)/(tn + fp)
    else:
         sensitivity = float(tp)/ (tp+fn)
         specificity = float(tn)/(tn + fp)
         MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    accuracy = np.float64(accuracy)
    sensitivity = np.float64(sensitivity)
    specificity = np.float64(specificity)
    MCC = np.float64(MCC)

    fpr, tpr, thresholds  =  roc_curve(testlabel, predict_y2)
    c=roc_auc_score(testlabel, predict_y2)    ##auc为Roc曲线下的面积
    precision, recall, thresholds = precision_recall_curve(testlabel, predict_y2)
    d = auc(recall, precision)    ##auc为pr曲线下的面积
    print("{:.5f} ".format(accuracy)+"{:.5f} ".format(c)+"{:.5f} ".format(d)+"{:.5f} ".format(sensitivity)+"{:.5f} ".format(specificity)+"{:.5f} ".format(MCC))

    acc.append(accuracy) 
    roc.append(c)
    pr.append(d) 
    sen.append(sensitivity)
    spe.append(specificity)
    Mcc.append(MCC)

    test_loss.append(eval[0])
    test_acc.append(eval[1])

p1 = np.savetxt('predict_test.txt',predict_test,fmt='%0.5f')
p2 = np.savetxt('testlabel_test.txt',testlabel_test,fmt='%0.1f')

print("acc:",acc)
print("roc:",roc)
print("pr：",pr)
print("sen：",sen)
print("spe：",spe)
print("Mcc：",Mcc)
print("Accuracy = {:.5f}".format(np.mean(acc)))
print("auroc = {:.5f}".format(np.mean(roc)))
print("auprc = {:.5f}".format(np.mean(pr)))
print("sensitivity = {:.5f}".format(np.mean(sen)))
print("specificity = {:.5f}".format(np.mean(spe)))
print("Mcc = {:.5f}".format(np.mean(Mcc)))

print("test_loss:",test_loss)
print("test_acc:",test_acc)
print("test_loss = {:.5f}".format(np.mean(test_loss)))
print("test_acc = {:.5f}".format(np.mean(test_acc)))

plt.figure(1) # 创建图表1
plt.title('Receiver operating characteristic Curve')# give plot a title
plt.plot(fpr, tpr, color='darkorange', label='mean ROC (area = {0:.5f})'.format(np.mean(roc)))
plt.xlabel('False Positive Rate') #横坐标是fpr
plt.ylabel('True Positive Rate')  #纵坐标是tpr
plt.legend(loc="lower right")
plt.savefig('roc.png')
plt.show()

plt.figure(2) # 创建图表2
plt.title('Precision-Recall Curve')# give plot a title
plt.plot(precision, recall, 'b', label='mean PR (area = {0:.5f})'.format(np.mean(pr)))
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.savefig('pr.png')
plt.show()



