import os
import sys
import numpy as np
import librosa
from scipy.io.wavfile import write as write_wav

import gin
import ddsp
import ddsp.training
from ddsp.training.models import Autoencoder
from ddsp.spectral_ops import compute_f0, compute_loudness
from pydub import AudioSegment
from pydub.generators import WhiteNoise

def save_audio(path, audio, sample_rate):
    """把浮点[-1,1]的音频保存成 16-bit WAV."""
    # 防止溢出
    audio = np.clip(audio, -1.0, 1.0)
    # 转 int16
    audio_int16 = (audio * 32767).astype(np.int16)
    write_wav(path, sample_rate, audio_int16)

# === 路径设置 ===
GIN_FILE   = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\models\flute\operative_config-0.gin'
CKPT_FILE  = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\models\flute\ckpt-20000'
INPUT_WAV  = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\audio\input2.wav'
OUTPUT_WAV = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output_flute\output2.wav'

# === 超参 ===
TARGET_SR  = 16000
FRAME_RATE = 250

# 解析配置时跳过那些不存在的绑定（SoloFlute、train.* 等）
gin.parse_config_file(GIN_FILE, skip_unknown=True)

# 1. 读并重采样
audio, sr = librosa.load(INPUT_WAV, sr=TARGET_SR, mono=True)
print(f"🎧 加载音频: {INPUT_WAV} 采样率={sr} 时长={len(audio)/sr:.2f}s")

# 2. 特征提取（手动做预处理）
f0_hz, f0_conf   = compute_f0(audio, FRAME_RATE)
loudness_db      = compute_loudness(audio, FRAME_RATE)
n = min(len(f0_hz), len(loudness_db))
f0_hz, f0_conf  = f0_hz[:n], f0_conf[:n]
loudness_db     = loudness_db[:n]
features = {
    'f0_hz': f0_hz[np.newaxis, :],
    'f0_confidence': f0_conf[np.newaxis, :],
    'loudness_db': loudness_db[np.newaxis, :]
}
print(f"📈 特征帧数: {n}")

# 3. 加载模型
gin.parse_config_file(GIN_FILE)
model = Autoencoder()
model.restore(CKPT_FILE)
print(f"✅ 模型加载: {CKPT_FILE}")

# 4. 推理合成
# 把 audio 和 特征 合并到同一个输入 dict
# 4. 推理合成
model_input = {
    'audio': audio[np.newaxis, :],
    'f0_hz': f0_hz[np.newaxis, :],
    'f0_confidence': f0_conf[np.newaxis, :],
    'loudness_db': loudness_db[np.newaxis, :]
}
outputs = model(model_input, training=False)

# 兼容单 Tensor 或 dict
if isinstance(outputs, dict):
    audio_tensor = outputs['audio_synth']
else:
    audio_tensor = outputs

audio_out = audio_tensor.numpy().flatten()

# 5. 保存
save_audio(OUTPUT_WAV, audio_out, TARGET_SR)

# 6.混响
# 读取输出的 .wav 文件
audio = AudioSegment.from_wav(OUTPUT_WAV)
# 模拟空间感的混响：尾部衰减 + 延迟叠加
def add_reverb_delay(audio, reverb_tail_ms=400, delay_ms=100, decay=0.4, repeats=4):
    # 创建延迟回声轨道
    echo = audio
    for i in range(1, repeats + 1):
        attenuated = audio - (i * decay * 10)  # 衰减
        echo = echo.overlay(attenuated, position=i * delay_ms)

    # 淡入淡出 + 尾部静音延伸
    reverbed = echo.fade_in(300).fade_out(reverb_tail_ms)
    return reverbed + AudioSegment.silent(duration=reverb_tail_ms)


# 添加效果
processed_audio = add_reverb_delay(audio)

# 保存处理后的音频
processed_path = OUTPUT_WAV.replace('.wav', '_reverb.wav')
processed_audio.export(processed_path, format='wav')
print(f"✨ 已保存带空间感的版本：{processed_path}")

from pydub import AudioSegment
import random

# 加载保存好的 DDSP 合成音频
original = AudioSegment.from_wav(OUTPUT_WAV)

# 3. 定义 granular_echo 函数
def granular_echo(audio, grain_ms=80, repeat=8, max_delay_ms=300):
    new_audio = audio
    for i in range(repeat):
        offset = random.randint(10, max_delay_ms)
        grain = audio[:grain_ms] - (i * 6)
        new_audio = new_audio.overlay(grain, position=offset * i)
    return new_audio

# 4. 应用 granular 效果
granular_version = granular_echo(original)

# 5. 保存带效果的版本
granular_path = OUTPUT_WAV.replace('.wav', '_fossilscape.wav')
granular_version.export(granular_path, format='wav')
print(f"✨ 已保存带颗粒感远古声音版本：{granular_path}")