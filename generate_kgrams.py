import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
k = 28 # size of grams 

def kgram(k, ts):
    composition = [ts[i:i+k] for i in range(0,len(ts)-k+1,k)]
    return composition

def weight_kgram(k, weights):
    composition = [weights[:, i:i+k] for i in range(0, len(weights[0])-k+1, k)]
    return composition
    
def calculate_product(w, x, b):
    '''
      Calculate the product of a sub-sequence of one time series and it's coefficient sub-matrix, w_i*x_i+b_i.
      Return the number of result less zeros.
    '''
    product = np.matmul(w, x)+b
    return len(product[product<0])

def main():
    # load one time series
    coffee_test = pd.read_csv('data/coffee_test.csv', sep=',', header=None).astype(float)
    coffee_test_y = coffee_test.loc[:, 0].astype(int)
    coffee_test_x = coffee_test.loc[:,1:]
    one_ts = np.array(coffee_test_x.loc[15,:])
    ts_grams = kgram(k, one_ts)

    # load coefficients
    coefficients = np.loadtxt('./inequality_coeff.txt', delimiter=',')
    weights = coefficients[:,:-2]
    bias = coefficients[:,286:-1]
    one_gram = ts_grams[0]
    one_gram = one_gram.reshape(-1, 28)

    print(weights.shape)
    print(weights[0].shape)
    weight_grams = weight_kgram(k, weights)
    print(weight_grams[1].shape)
#    bias_grams = kgram(k, bias)
 #   print(bias_grams[0].shape)
    num_less_zeros = {}
    for i in range(10):
        result = calculate_product(weight_grams[i], ts_grams[i], bias[i])
        num_less_zeros[i+1] = result
    print("number of less zeros: ", num_less_zeros)
    plot_tsgram(one_ts, ts_grams)

def plot_tsgram(ts,ts_grams):
    plt.plot(ts, 'b')
    plt.plot(range(168,196), ts_grams[6], 'r',linewidth=2.5)
    plt.plot(range(224,252), ts_grams[8], 'r', linewidth=2.5)
    plt.title("The shapelet")
    plt.show()
    
if __name__ == '__main__':
    main()
    
