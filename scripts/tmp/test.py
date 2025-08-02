import numpy as np
from pprint import pprint
import os
def possibleToReach(a, b):
    c = np.cbrt(a * b)
    re1 = a // c
    re2 = b // c
    pprint(c)
    print(re1,re2)
    os.system("pause")
    if (re1 * re1 * re2 == a) and (re2 * re2 * re1 == b):
        return True
    else:
        return False