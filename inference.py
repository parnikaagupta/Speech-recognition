import torch
import librosa
import numpy as np
import sys

# ---------- Emotion Labels ----------
emotion_labels = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# ---------- MFCC Extraction ----------
def extract_mfcc(file_path, max_len=130):
    y, sr = librosa.load(file_path, duration=3)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc = np.vstack([mfcc, delta, delta2])
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]

    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

# ---------- Model ----------
class EmotionCNNLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.lstm = torch.nn.LSTM(
            input_size=64 * 28,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = torch.nn.Linear(256, 8)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)

        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(out)

# ---------- Main ----------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    model = EmotionCNNLSTM()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    features = extract_mfcc(audio_path)

    with torch.no_grad():
        output = model(features)
        pred = torch.argmax(output, dim=1).item()

    print("Predicted Emotion:", emotion_labels[pred])
