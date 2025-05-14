import os
import glob
import math
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pysepm import pesq, stoi, composite  # pysepm functions
# You can also install soundfile if you wish to write audio, but it isn't needed here.

##################################
# CONFIGURATION & PARAMETERS
##################################
SAMPLE_RATE = 16000            # Model expects 16 kHz
TARGET_AUDIO_LENGTH = 2.0      # seconds (expected duration per sample)
TARGET_NUM_SAMPLES = int(SAMPLE_RATE * TARGET_AUDIO_LENGTH)

# Model parameters (must match training)
L = 5           # number of layers
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

# The model’s fixed input length:
FRAME_LENGTH = compute_valid_length(TARGET_NUM_SAMPLES, L, K, S_conv, RESAMPLE_FACTOR)

# Directories with test audio files (adjust these paths)
NOISY_TEST_DIR = r"C:\Users\Andrei\Desktop\Proiect Licenta\datasets\DS_10283_2791\noisy_testset_wav"
CLEAN_TEST_DIR = r"C:\Users\Andrei\Desktop\Proiect Licenta\datasets\DS_10283_2791\clean_testset_wav"

##################################
# MODEL DEFINITION: CausalDemucs
##################################
from model_def_gru import CausalDemucsSplit

##################################
# LOAD THE MODEL (best_model_ms_snsd.pth)
##################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CausalDemucsSplit(chin=1, chout=1, hidden=48, depth=5, kernel_size=K, stride=S_conv,
                     causal=True, resample=RESAMPLE_FACTOR, normalize=False, glu=True).to(device)
checkpoint = torch.load("best_model_ms_snsd_gru.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded for evaluation.")

#########################################
# UTILITY FUNCTIONS
#########################################
def pad_or_crop(audio, target_length):
    audio = np.atleast_1d(audio)  # Ensure audio is a numpy array with a shape.
    current_length = audio.shape[0]
    if current_length > target_length:
        return audio[:target_length]
    elif current_length < target_length:
        pad_amt = target_length - current_length
        return np.concatenate([audio, np.zeros((pad_amt,), dtype=audio.dtype)])
    else:
        return audio

#normalize the audio
def external_normalize(audio, eps=1e-3):
    """
    Normalize the audio externally.
    Returns:
      normalized_audio: audio divided by (std + eps)
      std: computed per-sample standard deviation (used later for re-scaling)
    """
    mono = audio.mean(dim=1, keepdim=True)
    std = mono.std(dim=-1, keepdim=True)
    normalized = audio / (std + eps)
    return normalized, std

def process_audio_segment(segment):
    """
    Process a 1D numpy array (of length FRAME_LENGTH) using the PyTorch model.
    Returns the enhanced audio as a 1D numpy array.
    """
    tensor_in = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    normalized, std = external_normalize(tensor_in)
    with torch.no_grad():
        enhanced = model(normalized)
    enhanced = enhanced * std
    enhanced = enhanced.squeeze().cpu().numpy()
    return enhanced


#########################################
# MAIN EVALUATION SCRIPT
#########################################
def main():
    # List test files; assumes matching files in both directories.
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
        # Load audio using torchaudio.
        noisy_waveform, sr_noisy = torchaudio.load(noisy_path)
        clean_waveform, sr_clean = torchaudio.load(clean_path)
        # Resample if needed.
        if sr_noisy != SAMPLE_RATE:
            noisy_waveform = torchaudio.transforms.Resample(sr_noisy, SAMPLE_RATE)(noisy_waveform)
        if sr_clean != SAMPLE_RATE:
            clean_waveform = torchaudio.transforms.Resample(sr_clean, SAMPLE_RATE)(clean_waveform)
        
        # Convert to mono.
        noisy_waveform = noisy_waveform.mean(dim=0).numpy()
        clean_waveform = clean_waveform.mean(dim=0).numpy()
        
        # Crop or pad to FRAME_LENGTH.
        noisy_segment = pad_or_crop(noisy_waveform, FRAME_LENGTH)
        clean_segment = pad_or_crop(clean_waveform, FRAME_LENGTH)
        
        # Process the noisy segment through the model.
        enhanced_segment = process_audio_segment(noisy_segment)
        
        # Compute PESQ (wideband).
        try:
            pesq_score = pesq(clean_segment, enhanced_segment, SAMPLE_RATE)
        except Exception as e:
            print(f"Error computing PESQ for {noisy_path}: {e}")
            pesq_score = np.nan
        
        # Compute STOI.
        try:
            stoi_score = stoi(clean_segment, enhanced_segment, SAMPLE_RATE)
        except Exception as e:
            print(f"Error computing STOI for {noisy_path}: {e}")
            stoi_score = np.nan
        
        # Compute composite metrics: composite returns (CSIG, CBAK, COVL)
        try:
            csig, cbak, covl = composite(clean_segment, enhanced_segment, SAMPLE_RATE)
        except Exception as e:
            print(f"Error computing composite metrics for {noisy_path}: {e}")
            csig, cbak, covl = (np.nan, np.nan, np.nan)
        
        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)
        csig_list.append(csig)
        cbak_list.append(cbak)
        covl_list.append(covl)
        
        print(f"File: {os.path.basename(noisy_path)} | PESQ: {pesq_score[1]:.3f}, STOI: {stoi_score:.3f}, "
              f"CSIG: {csig:.3f}, CBAK: {cbak:.3f}, COVL: {covl:.3f}")
    
    # Report average scores.
    avg_pesq = np.nanmean(pesq_scores)
    avg_stoi = np.nanmean(stoi_scores)
    avg_csig = np.nanmean(csig_list)
    avg_cbak = np.nanmean(cbak_list)
    avg_covl = np.nanmean(covl_list)
    
    print("\nEvaluation Complete:")
    print(f"Average PESQ: {avg_pesq:.3f}")
    print(f"Average STOI: {avg_stoi:.3f}")
    print(f"Average CSIG: {avg_csig:.3f}")
    print(f"Average CBAK: {avg_cbak:.3f}")
    print(f"Average COVL: {avg_covl:.3f}")

if __name__ == "__main__":
    main()