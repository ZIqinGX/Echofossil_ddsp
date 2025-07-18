# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
from scipy.io.wavfile import write as write_wav

import gin
from ddsp.training.models import Autoencoder

from pydub import AudioSegment
from pydub.generators import WhiteNoise
import random

# === Utility functions ===

def save_audio(path, audio, sample_rate):
    """Convert float[-1,1] audio to 16-bit WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    write_wav(path, sample_rate, audio_int16)

def granular_echo(audio: AudioSegment, grain_ms=80, repeat=8, max_delay_ms=300):
    new_audio = audio
    for i in range(repeat):
        offset = random.randint(10, max_delay_ms)
        grain = audio[:grain_ms] - (i * 6)
        new_audio = new_audio.overlay(grain, position=offset * i)
    return new_audio

def make_fossilscape(audio_path, output_base, modes):
    original = AudioSegment.from_wav(audio_path)
    os.makedirs(output_base, exist_ok=True)

    if "fossilscape_1" in modes:
        audio1 = granular_echo(original, grain_ms=60, repeat=10, max_delay_ms=250)
        audio1 = audio1.fade_in(300).fade_out(500)
        audio1 = audio1.low_pass_filter(1800)
        audio1.export(os.path.join(output_base, "fossilscape_1_cave_echo.wav"), format="wav")

    if "fossilscape_2" in modes:
        noise = WhiteNoise().to_audio_segment(duration=len(original)).fade_in(100).fade_out(200)
        audio2 = original.overlay(noise - 25)
        audio2 = audio2.low_pass_filter(1000)
        audio2 = granular_echo(audio2, grain_ms=50, repeat=6, max_delay_ms=200)
        audio2.export(os.path.join(output_base, "fossilscape_2_underwater_memory.wav"), format="wav")

    if "fossilscape_3" in modes:
        audio3 = original._spawn(
            original.raw_data,
            overrides={"frame_rate": int(original.frame_rate * 0.85)}
        ).set_frame_rate(original.frame_rate)
        audio3 = audio3.reverse()
        audio3 = granular_echo(audio3, grain_ms=100, repeat=7, max_delay_ms=300)
        audio3.export(os.path.join(output_base, "fossilscape_3_time_capsule.wav"), format="wav")

# === Configuration ===
GIN_FILE   = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\models\flute\operative_config-0.gin'
CKPT_FILE  = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\models\flute\ckpt-20000'
INPUT_WAV  = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\audio\input_stone.wav'
OUTPUT_WAV = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\output2.wav'
OUTPUT_DIR = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\processed'

TARGET_SR  = 16000
FRAME_RATE = 250  # 与模型 gin 文件一致

# === Load & Resample ===
audio, sr = librosa.load(INPUT_WAV, sr=TARGET_SR, mono=True)
print(f"🎧 Loaded audio: {INPUT_WAV}  SR={sr}  Duration={len(audio)/sr:.2f}s")

# === Feature Extraction ===
hop_size     = TARGET_SR // FRAME_RATE       # 16000//250 = 64 samples/frame
frame_length = hop_size * 2                  # 128 samples window

# 1) Loudness via RMS -> dB
rms          = librosa.feature.rms(
    y=audio,
    frame_length=frame_length,
    hop_length=hop_size
)[0]
loudness_db  = librosa.amplitude_to_db(rms, ref=np.max)
loudness_db  = np.nan_to_num(
    loudness_db,
    neginf=loudness_db[loudness_db != -np.inf].min() if np.any(loudness_db != -np.inf) else -60.0
)

# 2) Percussive/noise input: zero F0 & confidence
n_frames      = len(loudness_db)
f0_hz         = np.zeros(n_frames)
f0_confidence = np.zeros(n_frames)

print(f"📈 Feature frames: {n_frames}")

# === DDSP Synthesis ===
gin.parse_config_file(GIN_FILE, skip_unknown=True)
model = Autoencoder()
model.restore(CKPT_FILE)
print(f"✅ Model loaded: {CKPT_FILE}")

model_input = {
    'audio':          audio[np.newaxis, :],
    'f0_hz':          f0_hz[np.newaxis, :],
    'f0_confidence':  f0_confidence[np.newaxis, :],
    'loudness_db':    loudness_db[np.newaxis, :]
}
outputs = model(model_input, training=False)
audio_out = (outputs['audio_synth'] if isinstance(outputs, dict) else outputs).numpy().flatten()

# Save synthesized audio
save_audio(OUTPUT_WAV, audio_out, TARGET_SR)
print(f"✅ Synthesized audio saved: {OUTPUT_WAV}")

# === Post-Processing: Fossilscape Effects ===
make_fossilscape(
    OUTPUT_WAV,
    OUTPUT_DIR,
    ["fossilscape_1", "fossilscape_2", "fossilscape_3"]
)
print(f"✅ Fossilscape versions saved in: {OUTPUT_DIR}")
