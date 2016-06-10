import numpy
#C  = np.arange(0.1,10,0.1)
#G  = np.arange(0.1,10,0.1)
#EXP = [(c,gamma) for c in C for gamma in G]
from emotionDetection import main
import utils.defaults as defaults
import sys
samples = numpy.load(defaults.file_dataset)
labels = numpy.load(defaults.file_labels)
indx = numpy.load(defaults.model_feautures)

nInd=len(indx)
print "len",nInd
indx=indx[0:(nInd*0.4)]
print "len2",len(indx)
samples = samples[:,indx]
model = sys.argv[1]
if model=='knearest':
    ###############################################################
    # KNEAREST test
    bestKnearest = -1
    bestKnearestError = 100
    reps=50
    for i in xrange(1,6):
        totalError=0
        for _ in xrange(reps):
            filepath = ['--model',  'knearest', '--camera', 'off', '--param1', str(i), '--eval','n']
            totalError=totalError+main(filepath,samples, labels)
        
        knnError = float(totalError)/reps
        print "knn",i, 100-knnError
        if bestKnearestError>knnError:
            bestKnearestError=knnError
            bestKnearest=i
        #print "Mean Error k=%i: %f "%(i,knnError)
        
    print "Best knn: k=%i, accuracy:%f"%( bestKnearest, (100-bestKnearestError) ), "%"
elif model=='svm':
    ###############################################################
    ## SVM test 
    
    
    minErr = 100
    maxErr =-1
    sumErr = 0
    idxMin = -1
    reps=50
    for i in xrange(reps):
        fname = 'models/SVM/NU_SVR_RBF/svm%i.xml'%(i)
        print i,fname
        filepath = ['--model',  'svm', '--camera', 'off', '--eval','n']#, '--save',fname]
        currErr = main(filepath,samples, labels)
        sumErr=sumErr+currErr
        #print "err: ", (100-currErr)
        if currErr>maxErr:
            maxErr=currErr
        if currErr<minErr:
            minErr=currErr
            idxMin=i
            
    print "Best svm (i:%i)accuracy:%f, worst accuracy:%f, meanAccuracy:%f"%( idxMin, (100-minErr), (100-maxErr), (100-float(sumErr)/reps) )

elif model=='mlp':
    par_possible = [[46.0, 10.0], [15.0, 13.0], [31.0, 8.0], [12.0, 50.0], [24.0, 10.0], [27.0, 8.0], [5.0, 47.0], [21.0, 10.0], [36.0, 25.0], [60.0, 9.0], [62.0, 10.0], [69.0, 6.0], [9.0, 25.0], [10.0, 40.0], [17.0, 41.0], [25.0, 37.0], [55.0, 7.0], [57.0, 21.0], [65.0, 27.0], [66.0, 15.0]]

    par_possible = numpy.int8(numpy.asarray(par_possible))
    ###############################################################
    ## MLP test
    print "MLP parposible" 
    reps = 1
    err = []
    #for i,j in par_possible:
    for i in xrange(2,101):
        for j in xrange(2,101):
            totalError=0
            print i,j
            for _ in xrange(reps):
		        filepath = ['--model',  'mlp', '--camera', 'off', '--param1', str(i), '--param2', str(j), '--eval','n']
		        totalError=totalError+main(filepath,samples, labels)

            per = totalError/reps
            err.append([per, i, j])
            #print "nh1=%i: nh2:%i error:%f "%(i, j, 100-per)
    numpy.save('err_mlp_all.npy', err)
    print sorted(err)[0:20]
elif model=='boost':
    ###############################################################
    ## Boost test 
    reps = 50
    err = []

    for i in xrange(1,30):
        sumErr=0

        for _ in xrange(reps):
            filepath = ['--model',  'boost', '--camera', 'off', '--param1', str(i), '--eval','n']
            sumErr=sumErr+main(filepath,samples, labels)
        
        currErr  = float(sumErr)/ reps
        print i, currErrwhere
        err.append([currErr, i])    
        
    print sorted(err)[0:5]
    numpy.save('err_boost.npy', err)
    
elif model=='rtrees':
    ###############################################################
    ## Rtrees test 
    reps = 50
    err = []
    #xrange(50,350):
    for i in [217,86,91,230,218]:
        sumErr=0
        print i
        for _ in xrange(reps):
            filepath = ['--model',  'rtrees', '--camera', 'off', '--param1', str(i), '--eval','n']
            sumErr=sumErr+main(filepath,samples, labels)
        
        currErr  = float(sumErr)/ reps
        err.append([currErr, i])    
        
    print sorted(err)[0:5]
    numpy.save('err_rtrees3.npy', err)
