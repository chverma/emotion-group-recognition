import numpy as np
C  = np.arange(0.1,10,0.1)
G  = np.arange(0.1,10,0.1)
EXP = [(c,gamma) for c in C for gamma in G]
import subprocess
from emotionDetection import main
reps=20

for i in np.arange(1,30):
    totalError=0
    for _ in xrange(reps):
        print '-----------------------------------------------------'
        print '/blockmarker'
        #error: 18.92 %

        filepath = ['--model',  'knearest', '--camera', 'off', '--param1', str(i)]
        print filepath
        totalError=totalError+main(filepath)
    print "Mean Error k=%i: %f "%(i,totalError/reps)
    
    
exit()  
for (c, gamma) in EXP:
    print '-----------------------------------------------------'
    print '/blockmarker'
    #error: 18.92 %

    filepath = ['--model',  'svm', '--camera', 'off', '--param1', str(c),  '--param2', str(gamma)]
    print filepath
    main(filepath)
    
    
print "EXIT"
