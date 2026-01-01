import os
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import torch
import librosa
import numpy as np
import dataclasses
from typing import Any, Dict, List, Union

# ---- Dosya yolları ----
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA_PATH = os.path.join(BASE_PATH, "metadata.csv")

# ---- Model ayarları ----
MODEL_NAME = "openai/whisper-small"
SAMPLE_RATE = 16000

def prepare_dataset(metadata_path):
    df = pd.read_csv(metadata_path)
    
    # Convert relative paths to absolute paths
    df['path'] = df['path'].apply(lambda x: os.path.join(BASE_PATH, x))
    
    # Load audio files manually
    audio_data = []
    texts = []
    
    for _, row in df.iterrows():
        try:
            # Load audio with librosa
            audio_array, sr = librosa.load(row['path'], sr=SAMPLE_RATE)
            audio_data.append(audio_array)
            texts.append(row['text'])
        except Exception as e:
            print(f"Error loading {row['path']}: {e}")
            continue
    
    # Create dataset with audio arrays
    dataset_dict = {
        'audio': audio_data,
        'text': texts
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def preprocess(example, processor):
    audio_array = example["audio"]  # This is now a numpy array
    inputs = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    
    # Tokenize the text using text_target parameter
    labels = processor.tokenizer(text_target=example["text"], return_tensors="pt").input_ids
    
    # Remove batch dimension and convert to list
    inputs["input_features"] = inputs["input_features"].squeeze(0)
    inputs["labels"] = labels.squeeze(0)
    
    return inputs

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def main():
    print("Veri hazırlanıyor...")
    dataset = prepare_dataset(METADATA_PATH)

    print("Model ve tokenizer yükleniyor...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    dataset = dataset.map(lambda x: preprocess(x, processor), remove_columns=dataset.column_names)

    print("Model yükleniyor...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=10,
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=5,
        save_steps=50,
        eval_strategy="no",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    print("Eğitim başlıyor...")
    trainer.train()

if __name__ == "__main__":
    main()
