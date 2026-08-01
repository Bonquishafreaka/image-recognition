# Image Recognition

A simple image classifier with a web interface. Upload an image and it returns
the top 5 predictions with confidence scores.

Built on a pretrained MobileNetV2 model (trained on ImageNet's 1000 categories),
wrapped in a Gradio interface.

## Requirements

- Python 3.9+
- See `requirements.txt`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the app:

```bash
python app.py
```

Gradio will print a local URL. In GitHub Codespaces, a forwarded port
(usually 7860) will pop up — open it in the browser, then upload an image.

## How it works

- **Model:** Pretrained MobileNetV2 from torchvision. Weights download and
  cache automatically on first run.
- **Preprocessing:** Uses the exact transforms the model was trained with.
- **Output:** Softmax probabilities over 1000 ImageNet categories; the
  interface shows the top 5.

## Project structure

image-recognition/
├── requirements.txt # dependencies
├── app.py # the app + interface
└── README.md


## Roadmap

- [ ] Fine-tune on custom categories (transfer learning with a frozen backbone
      and a replaced classification head)
- [ ] Add a training script (`train.py`) to run on GPU (e.g. Google Colab)
- [ ] Load fine-tuned weights in `app.py`