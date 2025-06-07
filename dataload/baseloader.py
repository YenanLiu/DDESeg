# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import librosa
import torch
import torchaudio
import numpy as np
import math

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, args):
        self.sampling_rate = args.sample_rate
        self.target_len = int(args.audDur * args.sample_rate)

        self.snr_db = getattr(args, "snr_db", None)
    
    def _wav_repeat(self, wav):
        repeat_times = math.ceil(self.target_len / len(wav))
        extended_wav = np.tile(wav, repeat_times)

        # cutting to the target length
        extended_wav = extended_wav[:self.target_len]
        return extended_wav

    def _add_noise_with_snr(self, audio, snr_db):
        """
        add SNR noise
        :param audio: original audio signal (numpy array)
        :param snr_db: targeted snr noise (db)
        :return: noised audio signal
        """
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        noisy_audio = (audio + noise).astype(np.float32)
        return noisy_audio

    def _load_audio_file(self, path):
        if path.endswith(".mp3"):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)

            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
            else:
                audio_raw = audio_raw[:, 0]
        else:
            audio_raw, rate = librosa.load(path, mono=True)
        
        return audio_raw, rate

    def _load_audio(self, path):
        audio_raw, rate = self._load_audio_file(path)

        if rate != self.sampling_rate:
            audio_raw = librosa.resample(audio_raw, orig_sr=rate, target_sr=self.sampling_rate)

        audio_raw = self._wav_repeat(audio_raw)
 
        if self.snr_db is not None:
            audio_raw = self._add_noise_with_snr(audio_raw, self.snr_db)

        return audio_raw
    
    def _mix_audios(self, audios):
        N = len(audios)
        for n in range(N):
            audios[n] /= N

        audio_mix = np.asarray(audios).sum(axis=0)

        return torch.from_numpy(audio_mix)