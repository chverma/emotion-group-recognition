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

##IMAGES
img_directory = '/home/chverma/UPV/TFG/data/faces/union/'
neutral_imgs = img_directory+'neutral/'
happy_imgs = img_directory+'happy/'
disgust_imgs = img_directory+'disgust/'


fear_imgs = img_directory+'fear/'
surprised_imgs = img_directory+'surprised/'
sad_imgs = img_directory+'sad/'

##SAVED MODELS
model_svm_xml = 'models/svmDefault.xml'
model_knearest_xml = '/home/chverma/UPV/TFG/pythonDlibLendmark/models/emotion_KNearest.xml'
model_shape = 'models/shape_predictor_68_face_landmarks.dat'
