#!/usr/bin/env python3
"""
Encode a reference audio into codec tokens for voice cloning.

Uses Qwen3-TTS speech tokenizer to extract 16-group codec tokens
from a WAV file. These tokens are then used as prefix for the talker
to clone the voice characteristics.

Requirements:
  pip install torch numpy scipy
  pip install qwen-tts  # For speech tokenizer

Usage:
  python3 scripts/encode_reference_audio.py --audio reference.wav --output ref_codec_tokens.npy
"""

import os
import argparse
import time
import torch
import numpy as np
import wave as wavmod

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_wav(path):
    """Load WAV file to float32 tensor."""
    import scipy.io.wavfile as wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data).unsqueeze(0), sr


def main():
    parser = argparse.ArgumentParser(description="Encode reference audio to codec tokens")
    parser.add_argument("--audio", required=True, help="Reference audio WAV file")
    parser.add_argument("--model_dir", default=None, help="Path to Qwen3-TTS model")
    parser.add_argument("--output", default="ref_codec_tokens.npy")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (creates prompt_dir structure)")
    parser.add_argument("--ref_text", default=None,
                        help="Text spoken in the reference audio")
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/"
            "snapshots/c27fe8aa05b732b1376d0f6a1e522fbccb84abbd"
        )

    waveform, sr = load_wav(args.audio)
    print(f"Audio: {args.audio}")
    print(f"  Duration: {waveform.shape[1]/sr:.2f}s, SR: {sr}")

    st_dir = os.path.join(args.model_dir, "speech_tokenizer")
    print(f"\nLoading speech tokenizer...")

    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    t0 = time.time()
    tokenizer = Qwen3TTSTokenizer.from_pretrained(st_dir, torch_dtype=torch.float32)
    print(f"Tokenizer loaded in {time.time()-t0:.1f}s")

    print("\nEncoding audio...")
    t0 = time.time()
    with torch.no_grad():
        result = tokenizer.encode([waveform.numpy().flatten()], sr=sr)
    print(f"Encode time: {time.time()-t0:.2f}s")

    # Extract codec tokens
    if isinstance(result, dict):
        codes = result.get("audio_codes", None)
    elif isinstance(result, (list, tuple)):
        codes = result[0] if len(result) > 0 else None
    else:
        codes = result

    if codes is None:
        print("ERROR: No codes returned")
        return

    if isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()
    if isinstance(codes, list):
        codes = codes[0]
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

    if codes.ndim == 3:
        codes = codes[0]
    if codes.ndim == 2 and codes.shape[0] == 16:
        codes = codes.T

    n_tokens, n_groups = codes.shape
    print(f"Tokens: {n_tokens}, Groups: {n_groups}")
    print(f"Audio from tokens: {n_tokens/12:.2f}s")

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "ref_codec_tokens.npy")
        np.save(output_path, codes[:min(n_tokens, args.max_tokens)].astype(np.int64))
        if args.ref_text:
            with open(os.path.join(args.output_dir, "ref_text.txt"), "w") as f:
                f.write(args.ref_text)
        print(f"\nSaved prompt_dir: {args.output_dir}")
    else:
        np.save(args.output, codes[:min(n_tokens, args.max_tokens)].astype(np.int64))
        print(f"\nSaved: {args.output}")

    # Decode back for reference
    print("\nDecoding back to audio (PyTorch reference)...")
    actual_len = min(n_tokens, args.max_tokens)
    t0 = time.time()
    with torch.no_grad():
        decoded = tokenizer.decode([{"audio_codes": torch.from_numpy(codes[:actual_len])}])
    print(f"Decode time: {time.time()-t0:.2f}s")

    if isinstance(decoded, tuple):
        audio_out, fs = decoded
    else:
        audio_out = decoded
        fs = 24000

    if isinstance(audio_out, torch.Tensor):
        audio_np = audio_out.cpu().numpy().flatten()
    else:
        audio_np = np.array(audio_out).flatten()

    ref_wav = args.output.replace(".npy", "_decoded.wav")
    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    with wavmod.open(ref_wav, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_int16.tobytes())
    print(f"Saved reference decoded: {ref_wav}")


if __name__ == "__main__":
    main()
