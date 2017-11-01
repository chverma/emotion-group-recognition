import numpy
import utils.defaults as defaults

KDEF_data=numpy.load(defaults.KDEF_data)
KDEF_labels=numpy.load(defaults.KDEF_labels)

UNION_data=numpy.load(defaults.UNION_data)
UNION_labels=numpy.load(defaults.UNION_labels)

CAFE_data=numpy.load(defaults.CAFE_data)
CAFE_labels=numpy.load(defaults.CAFE_labels)


KDEF_data=numpy.vstack([KDEF_data,UNION_data])
KDEF_data=numpy.vstack([KDEF_data,CAFE_data])

print len(KDEF_labels), len(UNION_labels)
KDEF_labels=numpy.concatenate((KDEF_labels, UNION_labels), axis=0)
KDEF_labels=numpy.concatenate((KDEF_labels, CAFE_labels), axis=0)
print len(KDEF_labels)

numpy.save('dataset/mix_data.npy',KDEF_data)
numpy.save('dataset/mix_labels.npy',KDEF_labels)