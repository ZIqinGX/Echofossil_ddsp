# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
from scipy.signal import fftconvolve
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import random

# === 路径配置 ===
TARGET_SR   = 16000
INPUT_WAV   = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\audio\input_stone.wav'
DDSP_WAV    = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\processed\fossilscape_2_underwater_memory.wav'
IR_LIST     = [
    r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\ir\1a_marble_hall.wav',
    r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\ir\2b_mine.wav',
    r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\ir\3c_hm.wav'
]
OUTPUT_DIR      = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\schemeB'
MIXED_IR_WAV    = os.path.join(OUTPUT_DIR, 'mixed_ir_underwater.wav')
CONV_MIXED_WAV  = os.path.join(OUTPUT_DIR, 'mixed_conv_underwater.wav')
LOOPED_MIXED_WAV= os.path.join(OUTPUT_DIR, 'mixed_looped_underwater.wav')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. 加载输入和 IR ===
# 原始用于时长
orig, sr0   = librosa.load(INPUT_WAV, sr=TARGET_SR, mono=True)
# DDSP 输出
audio, sr1  = librosa.load(DDSP_WAV, sr=TARGET_SR, mono=True)
assert sr0 == sr1 == TARGET_SR, "采样率必须为 TARGET_SR"

# 加载所有 IR，并对齐长度
irs = []
max_len = 0
for path in IR_LIST:
    ir, sr2 = librosa.load(path, sr=TARGET_SR, mono=True)
    assert sr2 == TARGET_SR
    irs.append(ir)
    max_len = max(max_len, len(ir))

# === 2. 混合 IR ===
# 方式 A: 简单平均
irs_padded = [np.pad(ir, (0, max_len - len(ir))) for ir in irs]
mixed_ir   = np.mean(irs_padded, axis=0)

# 或者你可以用加权求和（比如 weights = [0.5,0.3,0.2]）
# weights = np.array([0.5,0.3,0.2])[:,None]
# mixed_ir = np.sum(np.stack(irs_padded) * weights, axis=0)

# 归一化
mixed_ir = mixed_ir / np.max(np.abs(mixed_ir))
# 保存混合 IR
write_wav(MIXED_IR_WAV, TARGET_SR, (mixed_ir * 32767).astype(np.int16))
print(f"✅ 混合 IR 已保存: {MIXED_IR_WAV}")

# === 3. 一次性卷积 ===
conv = fftconvolve(audio, mixed_ir, mode='full')
conv = conv / np.max(np.abs(conv))
write_wav(CONV_MIXED_WAV, TARGET_SR, (conv * 32767).astype(np.int16))
print(f"✅ 卷积混合 IR 完成并保存: {CONV_MIXED_WAV}, 时长≈{len(conv)/TARGET_SR:.2f}s")

# === 4. （可选）循环拼接到原始时长 ===
target_ms = int(len(orig) / TARGET_SR * 1000)
seg       = AudioSegment.from_wav(CONV_MIXED_WAV)
looped    = AudioSegment.empty()

while len(looped) < target_ms:
    cutoff = random.randint(800, 3000)
    seg_var = seg.low_pass_filter(cutoff)
    seg_var = seg_var.overlay(seg_var, position=random.randint(0, 200))
    looped += seg_var

looped = looped[:target_ms].fade_in(2000).fade_out(2000)
looped.export(LOOPED_MIXED_WAV, format='wav')
print(f"✅ 循环拼接并保存: {LOOPED_MIXED_WAV}, 时长≈{len(looped)/1000:.2f}s")
