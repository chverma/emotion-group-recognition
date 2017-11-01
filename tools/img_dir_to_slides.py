from PIL import Image
import sys
import utils.defaults as defaults
import os
if not len(sys.argv) == 2:
    raise SystemExit("Usage: %s src1 [src2] .. dest" % sys.argv[0])



filesToSlide=[]

for emotion in defaults.emotions:
        nfiles=0
        rootFiles='../data/faces/KDEF/'+emotion+'/'
        #rootFiles = '../database/georgia_faces/cropped_faces/happy/'

        files=[files for root, subdir, files in os.walk(rootFiles)]

        ex=True
        for f in files[0]:
            if emotion=='sad' and ex:
                ex=False
            else:
                nfiles=nfiles+1
                filesToSlide.append(rootFiles+f)
                if nfiles>=5:
                    break
        
            
images = map(Image.open, filesToSlide)
w = images[0].size[0]* 5
mh = max(i.size[1] for i in images)

result = Image.new("RGBA", (w, mh*7))

x = 0
y=0
for ind,i in enumerate(images):
    print "ind",ind
    if ind%5==0:
        y= mh*int(ind/5)
        x= 0
        print "y",y
    result.paste(i, (x, y))
    x += i.size[0]

result.save(sys.argv[1])
