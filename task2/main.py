from nn18_ex2_load import load_isolet
from BatchNormalizer import BatchNormalizer

def main():
    X, C, X_tst, C_tst = load_isolet()
    #print(X)

    print(X.shape)
    #print(C)
    #print(C.shape)

    b1 = BatchNormalizer(X)
    #print(b1.getMean()[0].shape)
    #print(b1.getStd()[0].shape)
    print(b1.getNormalizedClassBatches().shape)
    #b1.normalize()




if __name__ == '__main__':
     main()
