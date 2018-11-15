from nn18_ex2_load import load_isolet

def main():
    X, C, X_tst, C_tst = load_isolet()
    print(X)

    print(X.shape)
    print(C)
    print(C.shape)




if __name__ == '__main__':
     main()
