
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch


 -----------------------------
audio1_path = librosa.ex('trumpet')
audio2_path = librosa.ex('brahms')

y1, sr1 = librosa.load(audio1_path, sr=None)
y2, sr2 = librosa.load(audio2_path, sr=None)


 -----------------------------
target_sr = 16000
y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sr)
y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sr)


 -----------------------------
y1 = y1 / np.max(np.abs(y1))
y2 = y2 / np.max(np.abs(y2))


 -----------------------------
y1_trim, _ = librosa.effects.trim(y1)
y2_trim, _ = librosa.effects.trim(y2)


 -----------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
librosa.display.waveshow(y1, sr=target_sr)
plt.title("Raw Audio 1")

plt.subplot(2, 2, 2)
librosa.display.waveshow(y1_trim, sr=target_sr)
plt.title("Processed Audio 1")

plt.subplot(2, 2, 3)
librosa.display.waveshow(y2, sr=target_sr)
plt.title("Raw Audio 2")

plt.subplot(2, 2, 4)
librosa.display.waveshow(y2_trim, sr=target_sr)
plt.title("Processed Audio 2")

plt.tight_layout()
plt.show()


 -----------------------------
mfcc1 = librosa.feature.mfcc(y=y1_trim, sr=target_sr, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y2_trim, sr=target_sr, n_mfcc=13)


 -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
librosa.display.specshow(mfcc1, x_axis='time', sr=target_sr)
plt.title("MFCC - Audio 1")
plt.colorbar()

plt.subplot(1, 2, 2)
librosa.display.specshow(mfcc2, x_axis='time', sr=target_sr)
plt.title("MFCC - Audio 2")
plt.colorbar()

plt.tight_layout()
plt.show()

 -----------------------------
mfcc1_tensor = torch.tensor(mfcc1)
mfcc2_tensor = torch.tensor(mfcc2)

features_tensor = torch.stack([mfcc1_tensor, mfcc2_tensor])

print("Final Features Tensor Shape:", features_tensor.shape)
print("Feature Tensor Value Range:",
      features_tensor.min().item(), features_tensor.max().item())
