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
# 你的 DDSP 已生成的声景
DDSP_WAV    = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\processed\fossilscape_2_underwater_memory.wav'
# 串联使用的多个 IR
IR_LIST = [
    r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\ir\1a_marble_hall.wav',
    r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\ir\2b_mine.wav',
    r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\ir\3c_hm.wav'
]

OUTPUT_DIR     = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\schemeB'
CHAIN_CONV_WAV = os.path.join(OUTPUT_DIR, 'stone_multiple_conv.wav')
LOOPED_WAV     = os.path.join(OUTPUT_DIR, 'stone_multiple_looped.wav')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. 读取音频和 IR ===
# 原始用于确定时长
orig, sr0 = librosa.load(INPUT_WAV, sr=TARGET_SR, mono=True)
# DDSP 输出用于卷积
audio, sr1 = librosa.load(DDSP_WAV, sr=TARGET_SR, mono=True)
assert sr0 == sr1 == TARGET_SR, "采样率必须为 TARGET_SR"

# 依次读取 IR
irs = []
for ir_path in IR_LIST:
    ir, sr2 = librosa.load(ir_path, sr=TARGET_SR, mono=True)
    assert sr2 == TARGET_SR
    irs.append(ir)

print(f"🎧 原始时长 {len(orig)/TARGET_SR:.2f}s，DDSP 输出 {len(audio)/TARGET_SR:.2f}s，IR 数量：{len(irs)}")

# === 2. 串联卷积 ===
chain = audio.copy()
for idx, ir in enumerate(irs, start=1):
    chain = fftconvolve(chain, ir, mode='full')
    chain = chain / np.max(np.abs(chain))  # 每次卷积后归一化
    print(f"  卷积步骤 {idx} 完成，当前长度 {len(chain)/TARGET_SR:.2f}s")

# 保存链式卷积结果
write_wav(CHAIN_CONV_WAV, TARGET_SR, (chain * 32767).astype(np.int16))
print(f"✅ 已保存串联卷积文件: {CHAIN_CONV_WAV}")

# === 3. （可选）循环拼接到原始时长 ===
target_ms = int(len(orig) / TARGET_SR * 1000)
seg       = AudioSegment.from_wav(CHAIN_CONV_WAV)
looped    = AudioSegment.empty()

while len(looped) < target_ms:
    # 随机化微调
    cutoff = random.randint(800, 3000)
    seg_var = seg.low_pass_filter(cutoff)
    seg_var = seg_var.overlay(seg_var, position=random.randint(0, 200))
    looped += seg_var

looped = looped[:target_ms].fade_in(2000).fade_out(2000)
looped.export(LOOPED_WAV, format='wav')
print(f"✅ 已保存循环拼接文件: {LOOPED_WAV}，时长约 {len(looped)/1000:.2f}s")
