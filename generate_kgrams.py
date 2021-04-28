import pandas as pd
import numpy as np
def kgram(k, ts):
    composition = [ts[i:i+k] for i in range(0,len(ts)-k+1,k)]
    print(composition[0])
    print(composition[1])
    print(len(composition))
    

def main():
    # load one time series
    coffee_test = pd.read_csv('data/coffee_test.csv', sep=',', header=None).astype(float)
    coffee_test_y = coffee_test.loc[:, 0].astype(int)
    coffee_test_x = coffee_test.loc[:,1:]
    one_ts = np.array(coffee_test_x.loc[15,:])
    k = 10
    kgram(k, one_ts)
    
if __name__ == '__main__':
    main()
    
