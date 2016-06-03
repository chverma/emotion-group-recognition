##########DEFAULTS
dim = 2278 ## bacause I use 12 distances
dataset = 'KDEF' # JAFFE, KDEF, UNION
data_path = '/home/chverma/UPV/TFG/data/'

emotions=["happy","neutral","disgust","fear","surprised","sad","angry"]
CLASS_N = 7 ## At the moment 7 classes
    
use_log = True

## CSVs
happy_csv   = data_path+'csv/happy.csv'
neutral_csv = data_path+'csv/neutral.csv'
disgust_csv = data_path+'csv/disgust.csv'

fear_csv   = data_path+'csv/fear.csv'
surprised_csv = data_path+'csv/surprised.csv'
sad_csv = data_path+'csv/sad.csv'
angry_csv   = data_path+'csv/angry.csv'
## IMAGES
img_directory = data_path+'faces/'+dataset+"/"

neutral_imgs = img_directory+'neutral/'
happy_imgs = img_directory+'happy/'
disgust_imgs = img_directory+'disgust/'
angry_imgs = img_directory+'angry/'
fear_imgs = img_directory+'fear/'
surprised_imgs = img_directory+'surprised/'
sad_imgs = img_directory+'sad/'

## SAVED MODELS
model_svm_xml = 'models/svm12F.xml'
model_mlp_xml = 'models/mlp12F.xml'
model_knearest_xml = 'models/knearest1.xml'
model_boost_xml = 'models/boost6.xml'
model_shape = 'models/shape_predictor_68_face_landmarks.dat'

## DATASET (distances and labels)
### 68 punts, calcul de TOTES les distancies
CAFE_data = 'dataset/cafe_data.npy' 
CAFE_labels =  'dataset/cafe_labels.npy'

KDEF_data = 'dataset/kdef_data.npy' 
KDEF_labels =  'dataset/kdef_labels.npy'

UNION_data = 'dataset/union_data.npy' 
UNION_labels = 'dataset/union_labels.npy'

### VARIOS
union_1318_nolog = 'dataset/union_1318_nolog.npy' 
union_labels_1318_nolog =  'dataset/union_labels_1318_nolog.npy'

file_dataset12 = 'dataset/dataset12.npy' 
file_labels12 =  'dataset/labels12.npy'

file_dataset12NoLog = 'dataset/dataset12NoLog.npy' 
file_labels12NoLog =  'dataset/labels12NoLog.npy'

file_datasetKDEF = 'dataset/kdef_dataset12.npy' 
file_labelsKDEF =  'dataset/fdef_labels12.npy'

file_datasetKDEFNoLog = 'dataset/kdef_dataset12NoLog.npy' 
file_labelsKDEFNoLog =  'dataset/fdef_labels12NoLog.npy'

kdef_nolog_1318 = 'dataset/kdef_nolog_1318.npy'
kdef_labels_nolog_1318 = 'dataset/kdef_labels_nolog_1318.npy'

if dataset=='CAFE':
    file_dataset = CAFE_data
    file_labels = CAFE_labels
elif dataset=='JAFFE':
    file_dataset = JAFFE_data
    file_labels = JAFFE_labels
elif dataset=='KDEF':
    file_dataset = KDEF_data
    file_labels = KDEF_labels
elif dataset=='UNION':
    file_dataset = UNION_data
    file_labels = UNION_labels


