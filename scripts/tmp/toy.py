import numpy as np
def countbits(n):
    count = 0
    while n != 0:
        if n & 1:
            count += 1
        n = n // 2
    return count

def BitProduct(arr, N):
    product = 1
    for i in range(N):
        bits = countbits(arr[i])
        product *= bits
    return product

if __name__ == '__main__':
    arr = [3, 2, 4, 1, 5]
    N = len(arr)
    print(BitProduct(arr, N))
    print(np.prod(arr))