pip install librosa 
import librosa
pip install SoundFile
import soundfile 
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']
l =["01",'02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result
  #Load the data and extract features for each sound file
def load_data(test_size=0.25):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\waed\\Desktop\\speech-emotion-recognition_ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)
#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))
#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
filename = "C:\\Users\\waed\\Desktop\\speech-emotion-recognition_ravdess-data\\Actor_24\\03-01-03-01-01-02-24.wav"
data,sample_rate1 = librosa.load(filename, sr=22050, mono=True, offset=0.0, duration=50, res_type='kaiser_best')
librosa.display.waveplot(data,sr=sample_rate1, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000)
#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, 
                    epsilon=1e-08, hidden_layer_sizes=(300,), 
                    learning_rate='adaptive', max_iter=500)
#Train the model
model.fit(x_train,y_train)
# save the model to disk
filename = 'speech_emotion_model.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open('speech_emotion_model.sav', 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
#Predict for the test set
y_pred=model.predict(x_test)
#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
