# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
from scipy.signal import fftconvolve
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import random

# === 路径配置 ===
INPUT_WAV   = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\audio\input_stone.wav'
IR_WAV      = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\ir\2b_mine.wav'
DDSP_WAV    = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\processed\fossilscape_2_underwater_memory.wav'

OUTPUT_DIR  = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\schemeB'
CONV_WAV    = os.path.join(OUTPUT_DIR, 'stone_mine2b_conv.wav')
LOOPED_WAV  = os.path.join(OUTPUT_DIR, 'stone_mine2b_looped.wav')

TARGET_SR   = 16000  # 统一采样率
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. 读取音频 ===
# 1.1 原始石头录音，用于计算目标时长
orig, sr0 = librosa.load(INPUT_WAV, sr=TARGET_SR, mono=True)
assert sr0 == TARGET_SR

# 1.2 DDSP 后处理输出，用于卷积
audio, sr1 = librosa.load(DDSP_WAV, sr=TARGET_SR, mono=True)
# 1.3 脉冲响应 IR
ir,    sr2 = librosa.load(IR_WAV,    sr=TARGET_SR, mono=True)

assert sr1 == sr2 == TARGET_SR, "所有输入采样率必须等于 TARGET_SR"

print(f"🎧 原始时长: {len(orig)/sr0:.2f}s，DDSP 输出时长: {len(audio)/sr1:.2f}s，IR 时长: {len(ir)/sr2:.2f}s")

# === 2. FFT 卷积 ===
conv = fftconvolve(audio, ir, mode='full')
conv = conv / np.max(np.abs(conv))  # 归一化

# 保存一次性卷积结果
write_wav(CONV_WAV, TARGET_SR, (conv * 32767).astype(np.int16))
print(f"✅ 已保存卷积文件: {CONV_WAV}  时长: {len(conv)/TARGET_SR:.2f}s")

# === 3. 循环拼接到原始时长 ===
target_ms = int(len(orig) / TARGET_SR * 1000)

seg    = AudioSegment.from_wav(CONV_WAV)
looped = AudioSegment.empty()

while len(looped) < target_ms:
    # 随机低通 + 自叠加，增加每次拼接的变化
    cutoff = random.randint(800, 3000)
    seg_var = seg.low_pass_filter(cutoff)
    seg_var = seg_var.overlay(seg_var, position=random.randint(0, 200))
    looped += seg_var

# 裁剪到精确时长并淡入淡出
looped = looped[:target_ms]
looped = looped.fade_in(2000).fade_out(2000)

# 保存最终循环文件
looped.export(LOOPED_WAV, format='wav')
print(f"✅ 已保存循环拼接文件: {LOOPED_WAV}  时长: {len(looped)/1000:.2f}s")
