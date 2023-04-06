import librosa
import numpy
# read audio
def read_audio(filename, config):
	x, fs = librosa.load(filename, duration=config.duration, offset = config.offset)
	S = librosa.stft(x, N_FFT)
	p = np.angle(S)
	S = np.log1p(np.abs(S))  
	return S, fs

def data_generator(style_path, content_path, config):
    style_wv, style_sr = read_audio(style_path, config)
    content_wv, content_sr = read_audio(content_path, config)
    
    print("Sampling rate check...")
    if style_sr == content_sr:
        print("same.\n")
    else:
        print("reload the data for the same sampling rate...")
        exec("exit(0)")
        
    return style_wv, style_sr, content_wv, content_sr