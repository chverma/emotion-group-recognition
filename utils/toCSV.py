#!/bin/python

def parse(emotions, data_path, dataset):
    import glob
    csv_path = data_path+"csv/"
    img_path = data_path+"faces/"+dataset+"/"
    print csv_path
    print img_path
    for indx, item in enumerate(emotions):
        target = open(csv_path+item+".csv", 'w')
        list_= ""
        for filename in glob.glob(img_path+item+'/*'): #assuming gif
            list_+=filename.split("/")[-1]+"\t"
        target.write(list_[0:len(list_)-1])
        target.close()


