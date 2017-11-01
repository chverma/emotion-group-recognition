#!/usr/bin/env python

###ENRTRAR ACI http://www.shervinemami.info/faceRecognition.html
'''
     
'''

# built-in modules
from PIL import Image
import cv2
import cv
import math
import numpy
import time
import dlib
# models classes
from classes.SVM import SVM
from classes.KNearest import KNearest
from classes.MLP import MLP
from classes.RTrees import RTrees
from classes.Boost import Boost

# loaddata
from loaddata.LoadAndShuffleData import LoadAndShuffleData as loadData
from loaddata.processImage import getCamFrame
from loaddata.processImage import get_significant_points
from loaddata.processImage import get_distance
from loaddata.processImage import get_landmarks
from loaddata.processImage import annotate_landmarks
# defaults
import utils.defaults as defaults
from scripts.preprocess.drawPointsAndLines import drawPointsAndLines as drawPointsAndLines
indx=None
cols=None

def getLatexTable(TP, FN, FP, TN, TPR, SPC, PPV, NPV, FPR, FDR, FNR, ACC):
    float_formatter = lambda x: "%.2f" %( x*100)
    TPR=TPR*100
    #, SPC, PPV, NPV, FPR, FDR, FNR, ACC
    fileOut=open('prova.txt','w')
                  
    ### TP
    fileOut.write("%%% TP\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF} & svm & "+"\n".join([" & ".join( map(str, TP[0]) )])+
                  " \\\\\n& mlp & "+"\n".join([" & ".join( map(str, TP[1]) )])+
                  " \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{TP} & mix & "+"\n".join([" & ".join( map(str, TP[2]) )])+"\\\\ \hline")
    
    
    ### FN
    fileOut.write("\n\n%%% FN\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & "+"\n".join([" & ".join( map(str, numpy.asarray(FN[0])[0]) )])+
                  " \\\\\n& mlp & "+"\n".join([" & ".join( map(str, numpy.asarray(FN[1])[0]) )])+
                  " \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{FN} & mix & "+"\n".join([" & ".join( map(str, numpy.asarray(FN[2])[0]) )])+"\\\\ \hline")

    
    ### FP
    fileOut.write("\n\n%%% FP\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & "+"\n".join([" & ".join( map(str, numpy.asarray(FP[0])[0]) )])+
                  " \\\\\n& mlp & "+"\n".join([" & ".join( map(str, numpy.asarray(FP[1])[0]) )])+
                  " \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{FP} & mix & "+"\n".join([" & ".join( map(str, numpy.asarray(FP[2])[0]) )])+"\\\\ \hline")  

    ### TN
    fileOut.write("\n\n%%% TN\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & "+"\n".join([" & ".join( map(str, numpy.asarray(TN[0])[0]) )])+
                  " \\\\\n& mlp & "+"\n".join([" & ".join( map(str, numpy.asarray(TN[1])[0]) )])+
                  " \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{TN} & mix & "+"\n".join([" & ".join( map(str, numpy.asarray(TN[2])[0]) )])+"\\\\ \hline \hline")  
    
    ### TPR
    fileOut.write("\n\n%%% TPR\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(TPR[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(TPR[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{TPR} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter,numpy.asarray(TPR[2])[0]) )   ])+"}{}\\\\ \hline")  
    
    ### SPC
    fileOut.write("\n\n%%% SPC\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(SPC[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(SPC[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{SPC} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(SPC[2])[0]) )])+"}{}\\\\ \hline")  
    
    ### PPV
    fileOut.write("\n\n%%% PPV\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(PPV[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(PPV[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{PPV} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(PPV[2])[0]) )])+"}{}\\\\ \hline")  
    
    ### NPV
    fileOut.write("\n\n%%% NPV\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(NPV[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(NPV[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{NPV} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(NPV[2])[0]) )])+"}{}\\\\ \hline")  
    
    ### FPR
    fileOut.write("\n\n%%% FPR\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FPR[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FPR[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{FPR} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FPR[2])[0]) )])+"}{}\\\\ \hline")  
    
    ### FDR
    fileOut.write("\n\n%%% FDR\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FDR[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FDR[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{FDR} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FDR[2])[0]) )])+"}{}\\\\ \hline")  
    
    ### FNR
    fileOut.write("\n\n%%% FNR\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FNR[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FNR[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{FNR} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(FNR[2])[0]) )])+"}{}\\\\ \hline")  
    
    ### ACC
    fileOut.write("\n\n%%% ACC\n")
    fileOut.write("\\rowcolor[HTML]{EFEFEF}& svm & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(ACC[0])[0]) )])+
                  "}{} \\\\\n& mlp & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(ACC[1])[0]) )])+
                  "}{} \\\\\n\\rowcolor[HTML]{C4C3C3} \multirow{-3}{*}{ACC} & mix & \\SI{"+"\n".join(["}{} & \\SI{".join( map(float_formatter, numpy.asarray(ACC[2])[0]) )])+"}{}\\\\ \hline")  
    

    
def getVotes2(resSVM,resMLP):
    v=[]
    weightsSVM=[ .9852, .9596, .9987, .9293, .9505, .9270, 1 ]
    weightsMLP=[ .9825, .9735, .9964, .9036, .9743, .9446, .9969]
    for i in xrange(len(resSVM)):
        pSVM = weightsSVM[int(resSVM[i])]
        pMLP = weightsMLP[int(resMLP[i])]
        
        if pSVM>pMLP:
          v.append(resSVM[i])
        else:
          v.append(resMLP[i])
          
    return v
    
def getVotes(resSVM,resMLP,resSVM2,resMLP2,resFEAR, lab=[]):
    v=[]
    weightsSVM=[ .9852, .9596, .9987, .9293, .9505, .9270, 1 ]
    weightsMLP=[ .9825, .9735, .9964, .9036, .9743, .9446, .9969]
    counter2=0
    for i in xrange(len(resSVM)):
        votes=[0]*7
        votes[int(resSVM[i])]=votes[int(resSVM[i])]+1
        votes[int(resMLP[i])]=votes[int(resMLP[i])]+1
        votes[int(resSVM2[i])]=votes[int(resSVM2[i])]+1
        votes[int(resMLP2[i][0])]=votes[int(resMLP2[i][0])]+1
        m = max(votes)
        m=[k for k, j in enumerate(votes) if j == m]
        res=m[0]
        if len(m)>1:
            print "OOOOOOOSTIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            print votes
            print int(resSVM[i]),int(resMLP[i]),int(resSVM2[i]),int(resMLP2[i][0])
            print m, lab[i]
            import random
            res=m[random.randrange(0, 2, 1)]
            if lab[i]!=res:
                print "Acabe avent fet", counter2
  
            counter2=counter2+1
        v.append(res)

    return v 

def getVotesMLPModify(resSVM,resMLP,resSVM2,resMLP2, resFEAR, lab=[]):
    v=[]
    weightsSVM=[ .9852, .9596, .9987, .9293, .9505, .9270, 1 ]
    weightsMLP=[ .9825, .9735, .9964, .9036, .9743, .9446, .9969]
    count=0
    for i in xrange(len(resSVM)):
        
        if resSVM[i] in resMLP2[i]:
            res=resSVM[i]
            
            if resMLP2[i][0]==5: #tristesa
                res=5
            
            if res!=lab[i]:
                print "FIRST"
                print "jo dic: svm", resSVM[i]
                print "es", lab[i]
                print "i mlp te", resMLP2[i]
                print
            v.append(res)
        else:
            
            
            if resFEAR[i]==0:
                res=3
                print "canvie a fear"
            else:
                res = resSVM[i]
                
            if res!=lab[i]:
                print "OUT"
                print "jo dic", resSVM[i]
                print "es", lab[i]
                print "i mlp te",resMLP2[i]
                print "fear",resFEAR[i]
                print
            
            v.append(res)
            
    
    
    #exit()
    return v      
def predictFromCamera(model):
    win = dlib.image_window()
    ##predict from camera
    camera=cv2.VideoCapture(0)
    resp=0

    while True:
        
        img=getCamFrame(True,camera)
        cv2.putText(img, defaults.emotions[resp], (10,50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2, cv2.CV_AA)
        #b = cv.fromarray(img)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(b,'OpenCV',(10,500), font, 4,(255,255,255))#,2,cv2.LINE_AA)
        #img = numpy.asrray(b)
        landmark, obtained = get_landmarks(img)
        if obtained:
            significant_points = get_significant_points(landmark)
            distance_between_points =  get_distance(significant_points, defaults.use_log)
            
            #win.set_image(img)
            ##Print Points
            win.set_image(annotate_landmarks(img, numpy.matrix(landmark)))
            #dlib.hit_enter_to_continue()
            nDistance = len(distance_between_points)
            #print nDistance
            #print len(distance_between_points)
            if nDistance:
                
                ob = numpy.float32([distance_between_points])[:,indx]
                #print "1",len(ob[0])
                if cols!=None:
                    ob = numpy.delete(ob, list(cols), 1)
                #print "2",len(ob[0])
                resp = int(model.predict(ob)) # This predict does not work with Knearest
            #else:
            #    resp = int(model.predict(numpy.float32([distance_between_points])))
            #print "response",defaults.emotions[resp]           
def main(parameters, samples, labels):
    import getopt

    models = [SVM, MLP]

    models = dict( [(cls.__name__.lower(), cls) for cls in models] )
    '''
    print 'USAGE: emotionExperts.py [--camera <on/off> [--eval <y/n>] [--draw <y/n>]'
    print 'Models: ', ', '.join(models)
    print
    '''
    args, dummy = getopt.getopt(parameters, '', ['camera=', 'eval='])
    args = dict(args)
    args.setdefault('--camera', 'on')
    args.setdefault('--eval', 'y')
   
    modelSVM = SVM(None, None)
    modelSVM.set_params(None)
    modelSVM2 = SVM(None, None)
    modelSVM2.set_params(None)
    modelMLP = MLP(15, 76)
    modelMLP2 = MLP(15, 76)
    modelFearSVM = SVM(None, None)
    modelFearSVM.set_params(None,True)
    
    samples_train=None
    labels_train=None
    samples_test=None
    labels_test = None
    
    test_rateMix=0.
    test_rateSVM=0.
    test_rateMLP=0.
    test_rateFEAR=0.
    test_rateBOOST=0.
    cFea=0
    cErr=0
    confSVM=confFEAR=confMLP =confMIX= confBOOST=numpy.matrix(numpy.zeros((defaults.CLASS_N, defaults.CLASS_N), numpy.int32))
    for _ in xrange(100):
        from numpy import copy
        samples_train, labels_train, samples_test, labels_test  = loadData().shuffleData(samples, labels,.80)
        n_samples =  len(samples_train)
        samples_train2 = copy(samples_train[0:n_samples*0.7])
        labels_train2 = copy(labels_train[0:n_samples*0.7])
        
        samples_train3 = copy(samples_train[n_samples*0.3:-1])
        labels_train3 = copy(labels_train[n_samples*0.3:-1])

        #samples_train2 = copy(samples_train)
        #labels_train2 = copy(labels_train)
        
        labels_train_Fear = copy(labels_train)
        labels_train_Fear[labels_train_Fear != 3] = -1
        labels_train_Fear[labels_train_Fear == 3] = 0
        
        labels_test_Fear = copy(labels_test)
        labels_test_Fear[labels_test_Fear != 3] = -1
        labels_test_Fear[labels_test_Fear == 3] = 0
        
        modelSVM.train(samples_train3, labels_train3)
        modelSVM2.train(samples_train2, labels_train2)
        
        modelMLP.train(samples_train3, labels_train3)
        modelMLP2.train(samples_train2, labels_train2)
        modelFearSVM.train(samples_train, labels_train_Fear)
        #modelFearSVM.save("fear.xml")
        #modelBOOST.train(samples_train, labels_train)
        
        resSVM = modelSVM.predict(samples_test)
        resMLP = modelMLP.predict(samples_test)
        resSVM2 = modelSVM2.predict(samples_test)
        resMLP2 = modelMLP2.getActivationValues(samples_test,labels_test)

        resFEAR = modelFearSVM.predict(samples_test)
        #resBOOST= modelBOOST.getVotes(samples_test)
        
        resMIX = getVotes(resSVM,resMLP,resSVM2,resMLP2,resFEAR,labels_test)
        
        #test_rateBOOST  = test_rateBOOST+numpy.mean(numpy.asarray(resBOOST) != labels_test)
        test_rateMIX  = test_rateMix+numpy.mean(numpy.asarray(resMIX) != labels_test)
        test_rateSVM  = test_rateSVM+numpy.mean(resSVM != labels_test)
        test_rateFEAR  = test_rateFEAR+numpy.mean(resFEAR != labels_test_Fear)
        test_rateMLP  = test_rateMLP+numpy.mean(resMLP != labels_test)
        
        
        print "MIX"
        confMIX=confMIX+numpy.matrix(modelMLP.evaluate(None, labels_test, numpy.asarray(resMIX)))
        print "SVM"
        confSVM=confSVM+numpy.matrix(modelSVM.evaluate(None, labels_test, resSVM))
        print "FEAR"
        confFEAR=confFEAR+numpy.matrix(modelSVM.evaluate(None, labels_test_Fear, resFEAR))
        print "MLP"
        confMLP=confMLP+numpy.matrix(modelSVM.evaluate(None, labels_test, resMLP))
        #print "BOOST"
        #confBOOST=confBOOST+numpy.matrix(modelSVM.evaluate(None, labels_test, resBOOST))
        
    print 
    print "RESULTS SVM:"
    print confSVM
    
    
    print "TP_svm:"
    TP_svm=numpy.diagonal(confSVM)
    print TP_svm
    print
    
    print "FN_svm:"
    FN_svm=(confSVM.sum(axis=1).transpose()-TP_svm)[0]
    print FN_svm
    print
    
    print "FP_svm"
    FP_svm=confSVM.sum(axis=0)-TP_svm
    print FP_svm
    print

    print "TN_svm"
    TN_svm=confSVM.sum()*numpy.ones((1, defaults.CLASS_N), numpy.int32)-FP_svm-FN_svm-TP_svm
    print TN_svm
    print
    
    print "TPR or Sensitivity: TPR=TP/(TP+FN)"
    TPR_svm=numpy.float32(TP_svm)/(TP_svm+FN_svm)
    print TPR_svm
    print
    
    print "Specificity (SPC) or true negative rate (TNR): TPR=TN/(FP+TN)"
    SPC_svm=numpy.float32(TN_svm)/(TN_svm+FP_svm)
    print SPC_svm
    print
    
    print "precision or positive predictive value (PPV): PPV=TP/(TP+FP))"
    PPV_svm=numpy.float32(TP_svm)/(TP_svm+FP_svm)
    print PPV_svm
    print
    
    print "negative predictive value (NPV): NPV=TN/(TN+FN))"
    NPV_svm=numpy.float32(TN_svm)/(TN_svm+FN_svm)
    print NPV_svm
    print
    
    print "fall-out or false positive rate (FPR): FPR=1-SPC"
    FPR_svm= 1-SPC_svm
    print FPR_svm
    print
    
    print "false discovery rate (FDR) FDR=1-PPV"
    FDR_svm=1-PPV_svm
    print FDR_svm
    print
    
    print "miss rate or false negative rate (FNR) FNR=FN/(FN+TP)"
    FNR_svm=numpy.float32(FN_svm)/(FN_svm+TP_svm)
    print FNR_svm
    print
    
    print "accuracy (ACC) ACC=TP+TN/(TP+TN+FP+FN)"
    ACC_svm=numpy.float32(TP_svm+TN_svm)/(FN_svm+TP_svm+FP_svm+TN_svm)
    print ACC_svm
    print
    
    test_rateSVM=100-test_rateSVM/100
    print "test_rateSVM", test_rateSVM
    
    
    ##################################################
    ## MLP
    print 
    print "RESULTS MLP:"
    print confMLP
    
    
    print "TP:"
    TP_mlp=numpy.diagonal(confMLP)
    print TP_mlp
    print
    
    print "FN:"
    FN_mlp=(confMLP.sum(axis=1).transpose()-TP_mlp)[0]
    print FN_mlp
    print
    
    print "FP"
    FP_mlp=confMLP.sum(axis=0)-TP_mlp
    print FP_mlp
    print

    print "TN"
    TN_mlp=confMLP.sum()*numpy.ones((1, defaults.CLASS_N), numpy.int32)-FP_mlp-FN_mlp-TP_mlp
    print TN_mlp
    print
    
    print "TPR or Sensitivity: TPR=TP/(TP+FN)"
    TPR_mlp=numpy.float32(TP_mlp)/(TP_mlp+FN_mlp)
    print TPR_mlp
    print
    
    print "Specificity (SPC) or true negative rate (TNR): TPR=TN/(FP+TN)"
    SPC_mlp=numpy.float32(TN_mlp)/(TN_mlp+FP_mlp)
    print SPC_mlp
    print
    
    print "precision or positive predictive value (PPV): PPV=TP/(TP+FP))"
    PPV_mlp=numpy.float32(TP_mlp)/(TP_mlp+FP_mlp)
    print PPV_mlp
    print
    
    print "negative predictive value (NPV): NPV=TN/(TN+FN))"
    NPV_mlp=numpy.float32(TN_mlp)/(TN_mlp+FN_mlp)
    print NPV_mlp
    print
    
    print "fall-out or false positive rate (FPR): FPR=1-SPC"
    FPR_mlp= 1-SPC_mlp
    print FPR_mlp
    print
    
    print "false discovery rate (FDR) FDR=1-PPV"
    FDR_mlp=1-PPV_mlp
    print FDR_mlp
    print
    
    print "miss rate or false negative rate (FNR) FNR=FN/(FN+TP)"
    FNR_mlp=numpy.float32(FN_mlp)/(FN_mlp+TP_mlp)
    print FNR_mlp
    print
    
    print "accuracy (ACC) ACC=TP+TN/(TP+TN+FP+FN)"
    ACC_mlp=numpy.float32(TP_mlp+TN_mlp)/(FN_mlp+TP_mlp+FP_mlp+TN_mlp)
    print ACC_mlp
    print
    
    test_rateMLP=100-test_rateMLP/100
    print "test_rateMLP", test_rateMLP
    
    ##################################################
    ## MIX
    print 
    print "RESULTS MIX:"
    print confMIX
    
    
    print "TP:"
    TP_MIX=numpy.diagonal(confMIX)
    print TP_MIX
    print
    
    print "FN:"
    FN_MIX=(confMIX.sum(axis=1).transpose()-TP_MIX)[0]
    print FN_MIX
    print
    
    print "FP"
    FP_MIX=confMIX.sum(axis=0)-TP_MIX
    print FP_MIX
    print

    print "TN"
    TN_MIX=confMIX.sum()*numpy.ones((1, defaults.CLASS_N), numpy.int32)-FP_MIX-FN_MIX-TP_MIX
    print TN_MIX
    print
    
    print "TPR or Sensitivity: TPR=TP/(TP+FN)"
    TPR_MIX=numpy.float32(TP_MIX)/(TP_MIX+FN_MIX)
    print TPR_MIX
    print
    
    print "Specificity (SPC) or true negative rate (TNR): TPR=TN/(FP+TN)"
    SPC_MIX=numpy.float32(TN_MIX)/(TN_MIX+FP_MIX)
    print SPC_MIX
    print
    
    print "precision or positive predictive value (PPV): PPV=TP/(TP+FP))"
    PPV_MIX=numpy.float32(TP_MIX)/(TP_MIX+FP_MIX)
    print PPV_MIX
    print
    
    print "negative predictive value (NPV): NPV=TN/(TN+FN))"
    NPV_MIX=numpy.float32(TN_MIX)/(TN_MIX+FN_MIX)
    print NPV_MIX
    print
    
    print "fall-out or false positive rate (FPR): FPR=1-SPC"
    FPR_MIX= 1-SPC_MIX
    print FPR_MIX
    print
    
    print "false discovery rate (FDR) FDR=1-PPV"
    FDR_MIX=1-PPV_MIX
    print FDR_MIX
    print
    
    print "miss rate or false negative rate (FNR) FNR=FN/(FN+TP)"
    FNR_MIX=numpy.float32(FN_MIX)/(FN_MIX+TP_MIX)
    print FNR_MIX
    print
    
    print "accuracy (ACC) ACC=TP+TN/(TP+TN+FP+FN)"
    ACC_MIX=numpy.float32(TP_MIX+TN_MIX)/(FN_MIX+TP_MIX+FP_MIX+TN_MIX)
    print ACC_MIX
    print
    
    test_rateMIX=100-test_rateMIX/100
    #print " \\\\\n".join([" & ".join(map(str,line)) for line in a])
    print "test_rateMIX", test_rateMIX
    
    
    
    getLatexTable([TP_svm,TP_mlp,TP_MIX],[FN_svm,FN_mlp,FN_MIX], [FP_svm,FP_mlp,FP_MIX], [TN_svm,TN_mlp,TN_MIX], [TPR_svm,TPR_mlp,TPR_MIX], [SPC_svm,SPC_mlp,SPC_MIX], [PPV_svm,PPV_mlp,PPV_MIX],
                  [NPV_svm,NPV_mlp,NPV_MIX], [FPR_svm,FPR_mlp,FPR_MIX], [FDR_svm,FDR_mlp,FDR_MIX], [FNR_svm,FNR_mlp,FNR_MIX], [ACC_svm,ACC_mlp,ACC_MIX])
    if '--camera' in args:
        fn = args['--camera']
        if fn=='on': 
            predictFromCamera(model)
            
    return None
            
if __name__ == '__main__':
    import sys
    #print 'loading images from data file: npy'
    #If not exists, pass null

    samples = numpy.load(defaults.file_dataset)
    labels = numpy.load(defaults.file_labels)
    indx = numpy.load(defaults.model_feautures)
    nInd=len(indx)*0.4
    indx=indx[0:nInd]

    samples = samples[:,indx]
    nSamples= len(samples)
    if defaults.use_log:
        itemindex = numpy.where(samples==0)
        cols = set(itemindex[1])
        samples = numpy.delete(samples, list(cols), 1)
        samples= numpy.asarray(map(lambda x: math.log10(x), list(samples.reshape(-1,))), dtype=numpy.float32)
        samples = samples.reshape(nSamples,-1)
    '''
    samples=None
    labels=None
    '''
    #print("Total dataset size:")
    #print("n_samples: %d" % len(labels_train))
    #print("n_test: %d" % len(labels_test))
    main(sys.argv[1:], samples, labels)
    
        
    
