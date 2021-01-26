pip install sounddevice
pip install scipy
import sounddevice as sd
from scipy.io.wavfile import write
sp = 44100
s = 3
print ("recording...")
new_real_voice= sd.rec(int(s * sp), samplerate=sp, channels=2)
sd.wait() 
write('output2.wav', sp,new_real_voice)
