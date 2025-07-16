import os
import numpy as np
import librosa
import gin
from scipy.io.wavfile import write as write_wav

from ddsp.training.preprocessing import F0LoudnessPreprocessor
from ddsp.training.models import Autoencoder
from ddsp.spectral_ops import compute_f0, compute_loudness

# === 用户路径（请按需修改） ===
GIN_FILE   = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\models\violin\operative_config-0.gin'
CKPT_FILE  = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\models\violin\ckpt-40000'
INPUT_WAV  = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\audio\input.wav'
OUTPUT_WAV = r'C:\Users\OS\Desktop\MusicRecordingTransfeer\ddsp_test\output\output.wav'

# === 超参数 ===
TARGET_SR  = 16000  # 采样率
FRAME_RATE = 250    # 特征帧率

def save_audio(path: str, audio: np.ndarray, sr: int):
    """保存为 16-bit PCM WAV"""
    wav_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_wav(path, sr, wav_int16)
    print(f"✅ 合成文件保存到：{path}")

# 1. 载入 Gin 配置（只为了读取 model 的架构／超参）
gin.parse_config_file(GIN_FILE)

# 2. 读入并重采样音频
audio, sr = librosa.load(INPUT_WAV, sr=TARGET_SR, mono=True)
print(f"🎧 已加载音频：{INPUT_WAV} | SR={sr} | 时长={len(audio)/sr:.2f}s")

# 3. 手动提取原始特征
f0_hz, f0_conf = compute_f0(audio, FRAME_RATE)
loudness_db    = compute_loudness(audio, FRAME_RATE)

# 4. 对齐帧数到模型期望
hop_size = TARGET_SR // FRAME_RATE
n_frames = len(audio) // hop_size
f0_hz       = f0_hz[:n_frames]
f0_conf     = f0_conf[:n_frames]
loudness_db = loudness_db[:n_frames]
print(f"📈 特征帧数对齐：{n_frames}")

# 5. 初始化预处理器，只做“原始→scaled”，不重新计算 f0/loudness
prep = F0LoudnessPreprocessor(sample_rate=TARGET_SR, frame_rate=FRAME_RATE)
# 关闭内部重算
prep.compute_f0 = False
prep.compute_loudness = False

# 准备无 batch 的原始特征 dict
features_raw = {
    'f0_hz':         f0_hz[np.newaxis, :],
    'f0_confidence': f0_conf[np.newaxis, :],
    'loudness_db':   loudness_db[np.newaxis, :]
}

# 得到 scaled 特征（dict 包含 'f0_scaled' 和 'ld_scaled'）
scaled_feats = prep(features_raw, training=False)

# 6. 加载模型并 restore
model = Autoencoder()
model.restore(CKPT_FILE)
print(f"✅ 模型已加载：{CKPT_FILE}")

# 7. 直接 decode（传入 scaled_feats）
audio_tensor = model.decode(scaled_feats, training=False)
audio_out    = audio_tensor.numpy().flatten()

# 8. 保存合成结果
save_audio(OUTPUT_WAV, audio_out, TARGET_SR)
