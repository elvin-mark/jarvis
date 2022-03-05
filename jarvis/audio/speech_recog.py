import torch
import torchaudio
import os
import soundfile as sf

PATH_WAV2VEC2 = os.path.join(
    os.environ["DL_MODELS"], "wav2vec2.ckpt")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = ["<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R",
         "D", "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"]


class SpeechRecognition:
    def __init__(self):
        self.model = torchaudio.models.wav2vec2_base(aux_num_out=32)
        self.model.load_state_dict(torch.load(PATH_WAV2VEC2, map_location=dev))
        self.model = self.model.to(dev)
        self.model.eval()

    def transcript(self, audio_data):
        x = torch.from_numpy(audio_data).reshape(1, -1).float()
        with torch.no_grad():
            pred = self.model(x)[0]
        idxs = torch.argmax(pred, axis=2)[0].detach().cpu()
        idxs = torch.unique_consecutive(idxs).numpy()
        msg = "".join([vocab[i] for i in idxs]).replace(
            "|", " ").replace("<pad>", "")
        return msg


if __name__ == "__main__":
    sr = SpeechRecognition()
    data, rate = sf.read("samples/sample1.flac")
    print(sr.transcript(data))
