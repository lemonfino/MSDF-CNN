import os, os.path
import transint
import numpy as np

trainPath = '../CNN_Input/Train/'                       ##特征
testPath = '../CNN_Input/Test/'
trainlabelPath = '../CNN_Input/Trainlabel/'             ##标签（1或0）
testlabelPath = '../CNN_Input/Testlabel/'

rownum=500
txtType = 'txt'
txtLists = os.listdir(trainPath)  # 列出文件夹下所有的目录与文件
txtLists.sort()
trainrow=np.loadtxt('../CNN_Input/trainrow.txt')       ##数据集大小
testrow=np.loadtxt('../CNN_Input/testrow.txt')


def printall(path,row,judge):
    index = 0
    txtLists = os.listdir(path)
    txtLists.sort()
    allint=[1,2,3,4,5,6,7,8,9,10]
    for filename in txtLists:
        f = open(path + filename)
        a = transint.trans(f,row[index], judge)
        allint[index]=a
        index=index+1
    return(allint)


trainall=printall(trainPath,trainrow,rownum)
testall=printall(testPath,testrow,rownum)

trlabelall=printall(trainlabelPath,trainrow,1)
telabelall=printall(testlabelPath,testrow,1)

