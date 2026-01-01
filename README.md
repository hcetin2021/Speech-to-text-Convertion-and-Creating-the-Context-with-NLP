#  Whisper Transcription & NLP Analysis

A comprehensive Turkish speech recognition system with automatic transcription, intelligent summarization, and keyword extraction powered by fine-tuned Whisper models.

##  What It Does

- **Speech-to-Text**: Converts Turkish audio recordings to accurate text transcriptions
- **Smart Summarization**: Automatically generates concise summaries using sentence scoring
- **Keyword Extraction**: Identifies key topics using the YAKE algorithm
- **Web Interface**: User-friendly Gradio interface for easy interaction
- **Fine-tuning**: Train custom Whisper models on your own audio data

##  Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Launch Web App
cd whisper-finetuned
python app.py
```

Visit `http://127.0.0.1:7860` to use the interface.

## 🛠️ Technologies Used

### Core ML/AI
- **OpenAI Whisper** - Speech recognition model (fine-tuned small variant, 244M parameters)
- **Transformers** (Hugging Face) - Model loading and inference
- **PyTorch** - Deep learning framework

### NLP & Analysis
- **YAKE** - Unsupervised keyword extraction algorithm
- **NLTK** - Natural language processing utilities
- Custom sentence scoring for extractive summarization

### Audio Processing
- **librosa** - Audio analysis and feature extraction
- **soundfile** - Audio file I/O
- **torchaudio** - PyTorch audio processing

### Interface & Utils
- **Gradio** - Web UI framework
- **pandas** - Data handling for training metadata
- **numpy** - Numerical computations

## 📂 Project Structure

```
├── whisper-finetuned/
│   ├── app.py              # Web interface
│   ├── test_modelv4.py     # CLI analysis script
│   └── checkpoint-5/       # Trained model
├── fine_tune/
│   └── train.py            # Model training script
├── requirements.txt        # Dependencies
└── metadata.csv           # Training data labels
```

##  Key Features

### 1. Web Interface
- Drag-and-drop audio upload
- Adjustable summary length and keyword count
- Four analysis views: transcription, summary, keywords, statistics
- Support for multiple audio formats (WAV, MP3, etc.)

### 2. Advanced NLP Processing
- **Extractive Summarization**: Scores sentences by word importance and selects top N
- **YAKE Keywords**: Language-agnostic algorithm optimized for Turkish
- **Text Statistics**: Word count, vocabulary diversity, sentence metrics

### 3. Model Fine-tuning
- Custom training on Turkish audio datasets
- Seq2Seq training with mixed precision (FP16)
- Configurable hyperparameters (batch size, learning rate, epochs)

## Performance

- **Transcription**: Real-time processing for audio up to 30 seconds
- **Long Audio**: Automatic segmentation for longer recordings
- **Accuracy**: Optimized for Turkish speech recognition
- **Training Time**: ~1-2 minutes for small datasets (GPU recommended)

## 🔧 Usage Examples

### Python API
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

# Load model
processor = WhisperProcessor.from_pretrained("./whisper-finetuned/checkpoint-5")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-finetuned/checkpoint-5")

# Transcribe audio
audio, sr = sf.read("audio.wav")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
transcription = processor.batch_decode(
    model.generate(inputs.input_features), 
    skip_special_tokens=True
)[0]
```

### Command Line
```bash
cd whisper-finetuned
python test_modelv4.py  # Analyzes mennan1.wav and saves report
```

##  Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with 8GB+ VRAM (optional, for training)
- Windows/Linux/macOS

##  Training Custom Models

1. Add `.wav` files to `dosyalar/` folder
2. Create `metadata.csv` with paths and transcriptions
3. Run: `python fine_tune/train.py`
4. Model saved to `whisper-finetuned/checkpoint-5/`

## 🎓 Technical Details

- **Base Model**: `openai/whisper-small` (244M parameters)
- **Training**: Seq2Seq with gradient accumulation
- **Sample Rate**: 16kHz (auto-converted)
- **Segment Length**: 30-second chunks for long audio
- **Stopwords**: Turkish language optimized filtering

##  License

Uses the same license as OpenAI Whisper and Hugging Face Transformers.

##  Credits

Built with OpenAI Whisper, Hugging Face Transformers, and open-source audio processing tools.

---

**Made for Turkish speech recognition and NLP analysis** 🇹🇷
