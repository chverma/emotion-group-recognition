import numpy as np
elems = np.load('err_rtrees.npy')
lists  = []

for e, i in elems:
    lists.append([e, i])

print sorted(lists)[0:20]

