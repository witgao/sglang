import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
import logging
import torchaudio
import numpy as np
from einops import rearrange

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

from sglang.multimodal_gen.runtime.models.original import OriginalMelBandModel, OriginalWav2Vec2Model


def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


def loudness_norm(audio_array, sr=16000, lufs=-23):
    try:
        import pyloudnorm
    except:
        raise ImportError("pyloudnorm package is not installed")
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyloudnorm.normalize.loudness(
        audio_array, loudness, lufs)
    return normalized_audio


class AudioEncodeStage(PipelineStage):

    def __init__(self, audio_processor: OriginalMelBandModel, audio_encoder: OriginalWav2Vec2Model):
        super().__init__()
        self.audio_processor = audio_processor
        self.audio_encoder = audio_encoder

    def melband(self, model, audio, sample_rate):
        device = torch.device(torch.cuda.current_device())
        offload_device = torch.device("cpu")

        B, audio_channels, audio_length = audio.shape

        sr = 44100

        if audio_channels == 1:
            # Convert mono to stereo by duplicating the channel
            audio = audio.repeat(1, 2, 1)
            audio_channels = 2
            print("Converted mono input to stereo.")

        if sample_rate != sr:
            print(f"Resampling input {sample_rate} to {sr}")
            audio_np = audio.cpu().numpy()
            resampled = librosa.resample(
                audio_np, orig_sr=sample_rate, target_sr=sr, axis=-1)
            audio = torch.from_numpy(resampled)
        audio = original_audio = audio[0]

        C = 352800
        N = 2
        step = C // N
        fade_size = C // 10
        border = C - step

        if audio_length > 2 * border and border > 0:
            audio = F.pad(audio, (border, border), mode='reflect')

        windowing_array = get_windowing_array(C, fade_size, device)

        audio = audio.to(device)
        vocals = torch.zeros_like(audio, dtype=torch.float32).to(device)
        counter = torch.zeros_like(audio, dtype=torch.float32).to(device)

        total_length = audio.shape[1]
        num_chunks = (total_length + step - 1) // step

        model.to(device)

        # comfy_pbar = ProgressBar(num_chunks)

        for i in tqdm(range(0, total_length, step), desc="Processing chunks"):
            part = audio[:, i:i + C]
            length = part.shape[-1]
            if length < C:
                if length > C // 2 + 1:
                    part = F.pad(input=part, pad=(
                        0, C - length), mode='reflect')
                else:
                    part = F.pad(input=part, pad=(
                        0, C - length, 0, 0), mode='constant', value=0)

            x = model(part.unsqueeze(0))[0]

            window = windowing_array.clone()
            if i == 0:
                window[:fade_size] = 1
            elif i + C >= total_length:
                window[-fade_size:] = 1

            vocals[..., i:i+length] += x[..., :length] * window[..., :length]
            counter[..., i:i+length] += window[..., :length]
            # comfy_pbar.update(1)

        model.to(offload_device)

        estimated_sources = vocals / counter

        if audio_length > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]

        return (estimated_sources.unsqueeze(0).cpu(), sr)

    def wav2vec2(self, model, vocals, sample_rate, normalize_loudness, fps, num_frames):
        device = torch.device(torch.cuda.current_device())
        offload_device = torch.device("cpu")

        dtype = model.base_dtype
        wav2vec2 = model
        wav2vec2_feature_extractor = model.wav2vec_feature_extractor

        sr = 16000

        audio_input = vocals

        if sample_rate != 16000:
            audio_input = torchaudio.functional.resample(
                audio_input, sample_rate, sr)
        audio_input = audio_input[0][0]

        start_time = 0
        end_time = num_frames / fps

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        try:
            audio_segment = audio_input[start_sample:end_sample]
        except Exception:
            audio_segment = audio_input

        audio_segment = audio_segment.numpy()

        if normalize_loudness:
            audio_segment = loudness_norm(audio_segment, sr=sr)

        audio_feature = np.squeeze(
            wav2vec2_feature_extractor(
                audio_segment, sampling_rate=sr).input_values
        )

        audio_feature = torch.from_numpy(
            audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        audio_duration = len(audio_segment) / sr
        video_length = audio_duration * fps

        wav2vec2.to(device)
        embeddings = wav2vec2(audio_feature.to(dtype), seq_len=int(
            video_length), output_hidden_states=True)
        wav2vec2.to(offload_device)

        if len(embeddings) == 0:
            raise RuntimeError(
                "No valid audio embeddings extracted, please check inputs")

        audio_emb = torch.stack(
            embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        return audio_emb.cpu().detach()

    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:
        audio = batch.condition_audio
        sample_rate = batch.original_condition_audio_sample_rate

        vocals, sample_rate = self.melband(
            model=self.audio_processor, audio=audio, sample_rate=sample_rate)

        audio_embeds = self.wav2vec2(model=self.audio_encoder, vocals=vocals,
                                     sample_rate=sample_rate, normalize_loudness=True, fps=25, num_frames=25)
        batch.audio_embeds.append(audio_embeds)

        return batch
