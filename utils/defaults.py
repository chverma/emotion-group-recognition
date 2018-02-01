import os
'''DEFAULTS'''
dim = 2278  # because I use 12 distances
dataset = 'UNION'  # JAFFE, KDEF, UNION [is not Union is my dataset], CAFE
data_path = '<no_path>'
root_dir = os.path.join(os.path.abspath('.'), "")

emotions = ["happy", "neutral", "disgust", "fear", "surprised", "sad", "angry"]
CLASS_N = 7  # At the moment 7 classes

use_log = False

'''CSVs'''
happy_csv = data_path+'csv/happy.csv'
neutral_csv = data_path+'csv/neutral.csv'
disgust_csv = data_path+'csv/disgust.csv'

fear_csv = data_path+'csv/fear.csv'
surprised_csv = data_path+'csv/surprised.csv'
sad_csv = data_path+'csv/sad.csv'
angry_csv = data_path+'csv/angry.csv'
'''IMAGES'''
img_directory = data_path+'faces/'+dataset+"/"

neutral_imgs = img_directory+'neutral/'
happy_imgs = img_directory+'happy/'
disgust_imgs = img_directory+'disgust/'
angry_imgs = img_directory+'angry/'
fear_imgs = img_directory+'fear/'
surprised_imgs = img_directory+'surprised/'
sad_imgs = img_directory+'sad/'

'''SAVED MODELS'''
model_svm_xml = root_dir+'models/svm12F.xml'
model_mlp_xml = root_dir+'models/mlp12F.xml'
model_knearest_xml = root_dir+'models/knearest1.xml'
model_boost_xml = root_dir+'models/boost6.xml'
model_shape = root_dir+'models/shape_predictor_68_face_landmarks.dat'
model_feautures = root_dir+'models/KDEF_RFE.npy'
'''DATASET (distances and labels)'''
'''68 punts, calcul de TOTES les distancies'''
CAFE_data = root_dir+'dataset/cafe_data.npy'
CAFE_labels = root_dir+'dataset/cafe_labels.npy'

JAFFE_data = root_dir+'dataset/jaffe_data.npy'
JAFFE_labels = root_dir+'dataset/jaffe_labels.npy'

KDEF_data = root_dir+'dataset/kdef_data.npy'
KDEF_labels = root_dir+'dataset/kdef_labels.npy'

UNION_data = root_dir+'dataset/union_data.npy'
UNION_labels = root_dir+'dataset/union_labels.npy'

MIX_data = root_dir+'dataset/mix_data.npy'
MIX_labels = root_dir+'dataset/mix_labels.npy'

'''VARIOS'''
union_1318_nolog = root_dir+'dataset/union_1318_nolog.npy'
union_labels_1318_nolog = root_dir+'dataset/union_labels_1318_nolog.npy'

file_dataset12 = root_dir+'dataset/dataset12.npy'
file_labels12 = root_dir+'dataset/labels12.npy'

file_dataset12NoLog = root_dir+'dataset/dataset12NoLog.npy'
file_labels12NoLog = root_dir+'dataset/labels12NoLog.npy'

file_datasetKDEF = root_dir+'dataset/kdef_dataset12.npy'
file_labelsKDEF = root_dir+'dataset/fdef_labels12.npy'

file_datasetKDEFNoLog = root_dir+'dataset/kdef_dataset12NoLog.npy'
file_labelsKDEFNoLog = root_dir+'dataset/fdef_labels12NoLog.npy'

kdef_nolog_1318 = root_dir+'dataset/kdef_nolog_1318.npy'
kdef_labels_nolog_1318 = root_dir+'dataset/kdef_labels_nolog_1318.npy'

if dataset == 'CAFE':
    file_dataset = CAFE_data
    file_labels = CAFE_labels
elif dataset == 'JAFFE':
    file_dataset = JAFFE_data
    file_labels = JAFFE_labels
elif dataset == 'KDEF':
    file_dataset = KDEF_data
    file_labels = KDEF_labels
elif dataset == 'UNION':
    file_dataset = UNION_data
    file_labels = UNION_labels
elif dataset == 'MIX':
    file_dataset = MIX_data
    file_labels = MIX_labels
