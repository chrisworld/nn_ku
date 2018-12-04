from nn18_ex2_load import load_isolet
from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches

import numpy as np

def main():
    X, C, X_tst, C_tst = load_isolet()
    #print(X)
    Y=np.array([[1,2,3,4]])
    #print(X.shape)
    #print(C)
    #print(C.shape)


    b1 = BatchNormalizer(X,C,batch_size=40,shuffle=True)
    train = b1.getBatches(X,C)
    test  = b1.getBatches(X_tst,C_tst,test=True)
    #print(b1.cbatches.batch_num)

    print(X.shape)
    print(X_tst.shape)






    #b1.normalize()




if __name__ == '__main__':
     main()
