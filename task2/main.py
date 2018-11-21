from nn18_ex2_load import load_isolet
from BatchNormalizer import BatchNormalizer
import numpy as np

def main():
    X, C, X_tst, C_tst = load_isolet()
    #print(X)
    Y=np.array([[1,2,3,4]])
    print(X.shape)
    #print(C)
    #print(C.shape)

    b1 = BatchNormalizer(X)
    b1.getMean()
    b1.getStd()
    b1.getNormalized()
    b1.getBatches()



    #b1.normalize()




if __name__ == '__main__':
     main()
