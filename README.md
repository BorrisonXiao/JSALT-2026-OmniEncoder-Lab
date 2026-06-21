# JSALT 2026 — OmniEncoder Lab: CLIP, CLAP, and Audio-to-Image Retrieval

A hands-on tutorial notebook for the JSALT 2026 summer workshop. Students work with two contrastive multimodal models — [CLIP](https://openai.com/research/clip) (image–text) and [CLAP](https://github.com/LAION-AI/CLAP) (audio–text) — and combine them to build an audio-to-image retrieval system.

---

## Lab Structure

The lab is divided into three parts, each assigned to a subgroup of authors.

### Part 1 — Work with the CLIP Model
**Authors:** Cihan and Zihao

- Download CLIP (`openai/clip-vit-base-patch32`) and an ImageNet-style dataset (Imagenette by default, full ImageNet validation if available)
- Load and preprocess images and text
- Perform **zero-shot image classification** on a subset using templated captions (e.g., `"a photo of a {label}"`)
- Perform **text-to-image retrieval** over the image subset

### Part 2 — Work with the CLAP Model
**Authors:** Roger and Chin-Jou

- Download CLAP (`laion/clap-htsat-unfused`), ESC-50, and a LibriSpeech test subset from Hugging Face
- Load and preprocess audio and text
- Perform **zero-shot audio classification** on ESC-50 using templated captions
- Compare CLAP similarities between LibriSpeech waveforms and their paired transcriptions: for a given waveform, does CLAP assign the highest similarity to its correct transcript?

### Part 3 — Putting CLIP and CLAP Together
**Authors:** Jessica, Sandra, and Deb

**Goal:** Build an audio-to-image retrieval model. Given a waveform of an animal sound (dog bark, cat meow, etc.), return a picture of the correct animal.

Two approaches are compared:

1. **Direct retrieval (baseline):** Use the CLAP audio encoder to embed ESC-50 animal waveforms, then directly compare against CLIP image embeddings of ImageNet animal images using cosine similarity. Expected to perform poorly — CLAP and CLIP embeddings live in different spaces.

2. **Text-mediated retrieval:** Use CLAP's audio–text alignment to first predict the animal label from the sound, then feed that label as a text prompt into the CLIP text encoder to select the matching image embedding. Expected to perform significantly better.

Students are asked to explain why the two approaches differ in performance.

---

## Notebook

| File | Description |
|---|---|
| `JSALT_CLIP_CLAP_Lab_PART1_EXERCISE.ipynb` | Main lab notebook — Part 1 complete with solutions, Parts 2 and 3 have author placeholders |

The notebook is designed to run on **Google Colab with a GPU runtime** (T4 or better). Default subset sizes are kept small so cells run quickly while experimenting.

---

## Setup

Open the notebook in Colab, enable a GPU runtime (**Runtime → Change runtime type → GPU**), then run all cells from the top.

All required packages are installed in the first cell:

```python
!pip install -q torch torchvision torchaudio transformers datasets accelerate evaluate
!pip install -q pillow matplotlib numpy pandas scikit-learn tqdm librosa soundfile
```

No additional credentials are needed for the default dataset (Imagenette). To use full ImageNet validation, set `PART1_DATA_SOURCE = "hf_imagenet"` and ensure you have accepted the dataset terms on Hugging Face.

---

## Exercise Format

- Cells marked `### TODO` are student exercises.
- Each `### TODO` cell is followed by a collapsed `show solution` cell with a reference implementation.
- At the end of each part there are 4–5 discussion questions.
- A final reflection table at the end of the notebook summarizes results across all six systems.

---

## Datasets and Models

| Resource | Source | Notes |
|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | Hugging Face `transformers` |
| CLAP | `laion/clap-htsat-unfused` | Hugging Face `transformers` |
| Imagenette (default) | `frgfm/imagenette` | 10-class ImageNet-derived subset; no credentials needed |
| ImageNet validation | `ILSVRC/imagenet-1k` | Requires HF authentication / dataset acceptance |
| ESC-50 | `ashraq/esc50` | Environmental sound classification, 50 classes |
| LibriSpeech | `librispeech_asr` | English read speech; use `test-clean` split |

---

## Learning Objectives

By completing this lab, students will be able to:

1. Load pretrained CLIP and CLAP models and their processors.
2. Preprocess images, audio, and text for contrastive embedding models.
3. Use templated prompts for zero-shot classification.
4. Use cosine similarity for retrieval across text, image, and audio modalities.
5. Compare direct cross-model embedding retrieval with text-mediated retrieval.
6. Explain why two embedding spaces trained with different objectives may not be directly comparable.

---

## Contributing

Each assigned author group should fill in the `### TODO` cells and placeholder sections in their assigned part. Before the final release, confirm that:

- The notebook runs top-to-bottom on a fresh Colab GPU runtime.
- All default subsets fit within the tutorial time budget.
- Required authentication steps are documented before the first cell that needs them.
- The final reflection table can be completed from outputs already generated in the notebook.
