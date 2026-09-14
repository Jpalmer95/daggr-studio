# Brick registry verification

Generated: 2026-09-14T22:31:45+00:00
Verified with: gradio_client 2.7.0 / huggingface_hub 1.31.0

- tried: **44**
- running: **27**
- unusable: **17**

## Running bricks

| id | source | endpoint | modality | license | commercial |
|---|---|---|---|---|---|
| `chatterbox` | ResembleAI/Chatterbox | `/generate_tts_audio` | audio-tts | unknown | NO |
| `edge-tts` | innoai/Edge-TTS-Text-to-Speech | `/tts_interface` | audio-tts | gpl-2.0 | yes |
| `melo-tts` | mrfakename/MeloTTS | `/synthesize` | audio-tts | mit | yes |
| `bria-rmbg` | briaai/BRIA-RMBG-2.0 | `/image` | image-edit | unknown | NO |
| `kontext-dev` | black-forest-labs/FLUX.1-Kontext-dev | `/infer` | image-edit | mit | yes |
| `not-lain-bg-removal` | not-lain/background-removal | `/image` | image-edit | mit | yes |
| `qwen-image-edit` | Qwen/Qwen-Image-Edit | `/infer` | image-edit | unknown | NO |
| `flux1-dev` | black-forest-labs/FLUX.1-dev | `/infer` | image-gen | mit | yes |
| `flux1-schnell` | black-forest-labs/FLUX.1-schnell | `/infer` | image-gen | mit | yes |
| `sd35-large` | stabilityai/stable-diffusion-3.5-large | `/infer` | image-gen | other | NO |
| `z-image-turbo` | hf-applications/Z-Image-Turbo | `/generate_image` | image-gen | unknown | NO |
| `hunyuan3d-2` | Tencent/Hunyuan3D-2 | `/generation_all` | image-to-3d | unknown | NO |
| `hunyuan3d-21` | tencent/Hunyuan3D-2.1 | `/generation_all` | image-to-3d | unknown | NO |
| `trellis2` | microsoft/TRELLIS.2 | `/image_to_3d` | image-to-3d | mit | yes |
| `triposg` | VAST-AI/TripoSG | `/image_to_3d` | image-to-3d | unknown | NO |
| `en2fr-local` | abidlabs/en2fr | `/predict` | text | unknown | NO |
| `llm-deepseek-v3-2` | deepseek-ai/DeepSeek-V3.2 | `-` | text | mit | yes |
| `llm-deepseek-v4-1-flash` | deepseek-ai/DeepSeek-V4.1-Flash | `-` | text | mit | yes |
| `llm-gemma-3-12b` | google/gemma-3-12b-it | `-` | text | mit | yes |
| `llm-glm-4-7-flash` | zai-org/GLM-4.7-Flash | `-` | text | mit | yes |
| `llm-hermes-3-70b` | NousResearch/Hermes-3-Llama-3.1-70B | `-` | text | mit | yes |
| `llm-llama-3-1-8b` | meta-llama/Llama-3.1-8B-Instruct | `-` | text | mit | yes |
| `llm-llama-3-3-70b` | meta-llama/Llama-3.3-70B-Instruct | `-` | text | mit | yes |
| `llm-phi-4` | microsoft/phi-4 | `-` | text | mit | yes |
| `llm-qwen3-4b-2507` | Qwen/Qwen3-4B-Instruct-2507 | `-` | text | mit | yes |
| `shap-e` | hysts/Shap-E | `/image-to-3d` | text-to-3d | mit | yes |
| `ltx2-distilled` | Lightricks/ltx-2-distilled | `/generate_video` | video | unknown | NO |

## Unusable candidates (kept for provenance)

| id | source | error class |
|---|---|---|
| `ace-step` | ACE-Step/ACE-Step-v1-3.5B | RepositoryNotFoundError |
| `background-removal` | hf-applications/background-removal | execution probe failed |
| `depth-anything` | depth-anything/Depth-Anything-V2 | no usable endpoint among ['/on_submit'] |
| `image-caption-blip` | Salesforce/BLIP | ValueError |
| `kokoro-tts` | hexgrad/Kokoro-TTS | Space exposes no named API endpoints (Use via API is unavail |
| `llm-qwen3-8b` | Qwen/Qwen3-8B | RuntimeError |
| `moondream2` | vikhyatk/moondream2 | execution probe failed |
| `musicgen` | facebook/MusicGen | execution probe failed |
| `qwen3-tts` | ysharma/Qwen3-TTS | ValueError |
| `qwen3-vl` | Qwen/Qwen3-VL-8B-Instruct | RepositoryNotFoundError |
| `radames-real-time-text-to-image-sdxl-lightning` | radames/Real-Time-Text-to-Image-SDXL-Lightning | No callable endpoint found in view_api() |
| `real-esrgan` | ai-forever/Real-ESRGAN | RepositoryNotFoundError |
| `smolvlm` | HuggingFaceTB/SmolVLM-Instruct | RepositoryNotFoundError |
| `stable-audio` | stabilityai/stable-audio-open-1.0 | RepositoryNotFoundError |
| `swin2sr` | Xenova/swin2SR | RepositoryNotFoundError |
| `text-to-pokemon` | nateraw/text-to-pokemon | RepositoryNotFoundError |
| `wan21` | Wan-AI/Wan2.1 | declared video output but the endpoint returns only numbers  |

## Notes

Bricks that failed verification are kept in `spaces.json` with `status: error` so the
Medic can see them and substitute an alternative rather than retrying a dead Space.
Re-run `python scripts/verify_registry.py --refresh --only <id>` to retry a single brick.
