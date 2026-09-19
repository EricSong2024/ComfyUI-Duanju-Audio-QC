#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


TARGET_SR = 48_000


def split_prompts(text):
    return [item.strip() for item in text.replace("\n", ";").split(";") if item.strip()]


def load_audio(path):
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1)
    if sample_rate != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=TARGET_SR)
    return np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def technical_metrics(audio):
    eps = 1e-9
    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(np.square(audio)) + eps))
    rms_db = float(20.0 * math.log10(rms + eps))
    peak = float(np.max(abs_audio, initial=0.0))
    clipping_ratio = float(np.mean(abs_audio >= 0.999))
    silence_ratio = float(np.mean(abs_audio < 0.001))
    onset = librosa.onset.onset_strength(y=audio, sr=TARGET_SR)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset, sr=TARGET_SR, units="time")
    duration = max(len(audio) / TARGET_SR, eps)
    onset_rate = float(len(onset_times) / duration)

    penalty = 0.0
    if rms_db < -38.0 or rms_db > -7.0:
        penalty += 0.08
    if peak <= 1e-5:
        penalty += 1.0
    if clipping_ratio > 0.001:
        penalty += min(0.35, clipping_ratio * 30.0)
    if silence_ratio > 0.65:
        penalty += min(0.30, (silence_ratio - 0.65) * 0.8)
    if onset_rate > 5.0:
        penalty += min(0.20, (onset_rate - 5.0) * 0.025)
    return {
        "rms_db": round(rms_db, 4),
        "peak": round(peak, 6),
        "clipping_ratio": round(clipping_ratio, 8),
        "silence_ratio": round(silence_ratio, 6),
        "onset_rate_per_second": round(onset_rate, 4),
        "technical_penalty": round(penalty, 6),
    }


def normalized(features):
    return torch.nn.functional.normalize(features.float(), dim=-1)


def main():
    request_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    model_dir = request["model_dir"]
    audios = [load_audio(path) for path in request["audio_paths"]]
    positive = split_prompts(request["positive_text"])
    negative = split_prompts(request["negative_text"])
    if not positive:
        raise ValueError("positive_text must contain at least one prompt")

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to("cpu").eval()
    with torch.inference_mode():
        audio_inputs = processor(audios=audios, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        audio_features = normalized(model.get_audio_features(**audio_inputs))
        positive_inputs = processor(text=positive, return_tensors="pt", padding=True)
        positive_features = normalized(model.get_text_features(**positive_inputs))
        positive_scores = audio_features @ positive_features.T
        if negative:
            negative_inputs = processor(text=negative, return_tensors="pt", padding=True)
            negative_features = normalized(model.get_text_features(**negative_inputs))
            negative_scores = audio_features @ negative_features.T
        else:
            negative_scores = torch.zeros((len(audios), 1))

    candidates = []
    for index, audio in enumerate(audios):
        pos = float(torch.mean(positive_scores[index]).item())
        neg = float(torch.max(negative_scores[index]).item()) if negative else 0.0
        metrics = technical_metrics(audio)
        margin = pos - neg
        score = margin - float(metrics["technical_penalty"])
        candidates.append(
            {
                "index": index,
                "positive_score": round(pos, 6),
                "negative_score": round(neg, 6),
                "semantic_margin": round(margin, 6),
                "final_score": round(score, 6),
                "technical": metrics,
            }
        )

    selected = max(candidates, key=lambda item: item["final_score"])
    minimum = float(request["minimum_margin"])
    use_bgm = selected["semantic_margin"] >= minimum and selected["technical"]["technical_penalty"] < 0.5
    report = {
        "model": str(model_dir),
        "candidate_count": len(candidates),
        "minimum_margin": minimum,
        "use_bgm": use_bgm,
        "selected_index": selected["index"] if use_bgm else -1,
        "selected_score": selected["final_score"],
        "candidates": candidates,
    }
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
