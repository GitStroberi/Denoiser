import os
import glob
import math
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from pysepm import pesq, stoi, composite

# CONFIGURATION & PARAMETERS
SAMPLE_RATE = 16000            # Model expects 16 kHz
TARGET_AUDIO_LENGTH = 2.0      # seconds
TARGET_NUM_SAMPLES = int(SAMPLE_RATE * TARGET_AUDIO_LENGTH)

# Model architecture params (must match training)
L = 5           # number of encoder/decoder layers
K = 8           # kernel size
S_conv = 4      # stride
RESAMPLE_FACTOR = 4

def compute_valid_length(base_length, depth, kernel_size, stride, resample):
    L_val = math.ceil(base_length * resample)
    for _ in range(depth):
        L_val = math.ceil((L_val - kernel_size) / stride) + 1
        L_val = max(L_val, 1)
    for _ in range(depth):
        L_val = (L_val - 1) * stride + kernel_size
    return int(math.ceil(L_val / resample))

# Fixed input length for ONNX model
FRAME_LENGTH = compute_valid_length(TARGET_NUM_SAMPLES, L, K, S_conv, RESAMPLE_FACTOR)

# Directories for test data (adjust paths)
NOISY_TEST_DIR = r"C:\Users\Andrei\Desktop\Proiect Licenta\datasets\DS_10283_2791\noisy_testset_wav"
CLEAN_TEST_DIR = r"C:\Users\Andrei\Desktop\Proiect Licenta\datasets\DS_10283_2791\clean_testset_wav"

# Load ONNX model
ORT_SESSION = None

def load_onnx_model(onnx_path: str):
    global ORT_SESSION
    ORT_SESSION = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX model from {onnx_path}")

# Pad or crop audio to desired length
def pad_or_crop(audio: np.ndarray, target_length: int) -> np.ndarray:
    current_length = audio.shape[0]
    if current_length > target_length:
        return audio[:target_length]
    elif current_length < target_length:
        pad_amt = target_length - current_length
        return np.concatenate([audio, np.zeros((pad_amt,), dtype=audio.dtype)])
    return audio

# External normalization (match model's normalize=False behavior)
def external_normalize(audio: np.ndarray, eps: float = 1e-3):
    # audio: shape (1, length) or (length,)
    mono = audio  # already mono
    std = np.std(mono, keepdims=True)
    normalized = mono / (std + eps)
    return normalized.astype(np.float32), std.astype(np.float32)

# Run inference using ONNX Runtime
def process_segment_onnx(segment: np.ndarray) -> np.ndarray:
    # segment: 1D numpy array length FRAME_LENGTH
    # Normalize
    segment = segment.astype(np.float32)[None, :]  # shape (1, length)
    normalized, std = external_normalize(segment)
    # Prepare input: (1,1,FRAME_LENGTH)
    inp = normalized[None, :, :]
    # Run
    outputs = ORT_SESSION.run(None, {"audio": inp})
    enhanced = outputs[0]  # shape (1,1,FRAME_LENGTH)
    # Denormalize
    enhanced = enhanced.squeeze(0).squeeze(0)
    return (enhanced * std.squeeze()).astype(np.float32)

# MAIN EVALUATION
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ONNX speech enhancement model.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model file")
    args = parser.parse_args()

    load_onnx_model(args.onnx)

    noisy_files = sorted(glob.glob(os.path.join(NOISY_TEST_DIR, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(CLEAN_TEST_DIR, "*.wav")))
    if len(noisy_files) != len(clean_files):
        raise ValueError("Mismatch between number of noisy and clean files!")

    pesq_scores = []
    stoi_scores = []
    csig_list = []
    cbak_list = []
    covl_list = []

    for noisy_path, clean_path in zip(noisy_files, clean_files):
        noisy_wav, sr1 = torchaudio.load(noisy_path)
        clean_wav, sr2 = torchaudio.load(clean_path)
        # Resample if needed
        if sr1 != SAMPLE_RATE:
            noisy_wav = torchaudio.transforms.Resample(sr1, SAMPLE_RATE)(noisy_wav)
        if sr2 != SAMPLE_RATE:
            clean_wav = torchaudio.transforms.Resample(sr2, SAMPLE_RATE)(clean_wav)

        # Convert to mono
        noisy = noisy_wav.mean(dim=0).numpy()
        clean = clean_wav.mean(dim=0).numpy()

        # Pad/crop
        noisy_seg = pad_or_crop(noisy, FRAME_LENGTH)
        clean_seg = pad_or_crop(clean, FRAME_LENGTH)

        # Enhance
        enhanced_seg = process_segment_onnx(noisy_seg)

        # Metrics
        try:
            pesq_score = pesq(clean_seg, enhanced_seg, SAMPLE_RATE)
        except Exception:
            pesq_score = np.nan
        try:
            stoi_score = stoi(clean_seg, enhanced_seg, SAMPLE_RATE)
        except Exception:
            stoi_score = np.nan
        try:
            csig, cbak, covl = composite(clean_seg, enhanced_seg, SAMPLE_RATE)
        except Exception:
            csig, cbak, covl = (np.nan, np.nan, np.nan)

        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)
        csig_list.append(csig)
        cbak_list.append(cbak)
        covl_list.append(covl)

        print(f"{os.path.basename(noisy_path)} | PESQ: {pesq_score[1] if isinstance(pesq_score, tuple) else pesq_score:.3f}, "
              f"STOI: {stoi_score:.3f}, CSIG: {csig:.3f}, CBAK: {cbak:.3f}, COVL: {covl:.3f}")

    # Averages
    print("\n=== Average Scores ===")
    print(f"PESQ: {np.nanmean([s[1] if isinstance(s, tuple) else s for s in pesq_scores]):.3f}")
    print(f"STOI: {np.nanmean(stoi_scores):.3f}")
    print(f"CSIG: {np.nanmean(csig_list):.3f}")
    print(f"CBAK: {np.nanmean(cbak_list):.3f}")
    print(f"COVL: {np.nanmean(covl_list):.3f}")

if __name__ == "__main__":
    main()