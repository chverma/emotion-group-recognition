#python experimentSVM.py > experimentSVM.log
#grep 'error: [0-9][0-9].[0-9][0-9] %' experimentSVM.log > exp.log
f = open('exp.log')
import string
out = f.read()
tt = string.maketrans("\n"," ")
err = out.translate(tt).split(" ")

values = [float(err[i]) for i in xrange(1,len(err),3) if float(err[i])<40]
print values
