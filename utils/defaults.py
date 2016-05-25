##########DEFAULTS
dim = 12 ## bacause I use 12 distances


emotions=["happy","neutral","disgust","fear","surprised","sad"]
CLASS_N = 6 ## At the moment 6 classes
## CSVs
happy_csv   = '/home/chverma/UPV/TFG/data/csv/happy.csv'
neutral_csv = '/home/chverma/UPV/TFG/data/csv/neutral.csv'
disgust_csv = '/home/chverma/UPV/TFG/data/csv/disgust.csv'

fear_csv   = '/home/chverma/UPV/TFG/data/csv/fear.csv'
surprised_csv = '/home/chverma/UPV/TFG/data/csv/surprised.csv'
sad_csv = '/home/chverma/UPV/TFG/data/csv/sad.csv'

## IMAGES
img_directory = '/home/chverma/UPV/TFG/data/faces/union/'
#img_directory = '/home/chverma/UPV/TFG/data/faces/KDEF_Class/'
neutral_imgs = img_directory+'neutral/'
happy_imgs = img_directory+'happy/'
disgust_imgs = img_directory+'disgust/'

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
file_dataset51 = 'dataset/dataset51.npy' 
file_labels51 =  'dataset/labels51.npy'

file_dataset12 = 'dataset/dataset12.npy' 
file_labels12 =  'dataset/labels12.npy'

file_dataset12NoLog = 'dataset/dataset12NoLog.npy' 
file_labels12NoLog =  'dataset/labels12NoLog.npy'

file_datasetKDEF = 'dataset/kdef_dataset12.npy' 
file_labelsKDEF =  'dataset/fdef_labels12.npy'

file_datasetKDEFNoLog = 'dataset/kdef_dataset12NoLog.npy' 
file_labelsKDEFNoLog =  'dataset/fdef_labels12NoLog.npy'

file_dataset = file_dataset12
file_labels = file_labels12
use_log = True

