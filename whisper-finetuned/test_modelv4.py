from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import soundfile as sf
import yake
import re
from collections import Counter

# Model ve tokenizer yolları
model_dir = "./checkpoint-5"
audio_path = "chunk_1.wav"

# Model ve processor yükle
print("Model yükleniyor...")
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)

# Ses dosyasını oku
audio_input, sample_rate = sf.read(audio_path)
# Stereo ise mono'ya çevir
if len(audio_input.shape) > 1:
    audio_input = audio_input.mean(axis=1)
# Eğer örnekleme hızı 16000 değilse yeniden örnekle
target_sample_rate = 16000
if sample_rate != target_sample_rate:
    from scipy.signal import resample
    import numpy as np
    num_samples = int(len(audio_input) * target_sample_rate / sample_rate)
    audio_input = resample(audio_input, num_samples)
    sample_rate = target_sample_rate

# 30 saniyelik segmentlere böl ve sırayla transkribe et
segment_duration = 30  # saniye
segment_samples = target_sample_rate * segment_duration
total_samples = len(audio_input)
total_duration = total_samples / target_sample_rate

print(f"Toplam ses uzunluğu: {total_duration:.2f} saniye")
print(f"Segment sayısı: {int(total_samples / segment_samples) + (1 if total_samples % segment_samples != 0 else 0)}")
print("="*70)

all_transcriptions = []
for i in range(0, total_samples, segment_samples):
    start = i
    end = min(i + segment_samples, total_samples)
    segment = audio_input[start:end]
    
    if len(segment) == 0:
        continue
        
    print(f"İşleniyor: {start/target_sample_rate:.1f}-{end/target_sample_rate:.1f} saniye arası...")
    
    inputs = processor(segment, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    all_transcriptions.append(transcription)

# Tüm transkripsiyon metnini birleştir
full_text = " ".join(all_transcriptions)

print("\n" + "="*70)
print("TRANSKRİPSİYON:")
print("="*70)
for idx, t in enumerate(all_transcriptions):
    print(f"[{idx+1}. parça] {t}")

# NLP İşlemleri - Gelişmiş Versiyon
print("\n" + "="*70)
print("GELİŞMİŞ NLP ANALİZİ:")
print("="*70)

# 1. Cümle Skorlamalı Özet (Extractive Summarization)
def advanced_extract_summary(text, num_sentences=2):
    """Cümleleri skorlayarak en önemli cümleleri seç"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    if len(sentences) <= num_sentences:
        return ". ".join(sentences) + "."
    
    # Her cümlenin skorunu hesapla (kelime frekansı bazlı)
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    
    # Türkçe stopwords
    stopwords = {
        've', 'veya', 'ile', 'bu', 'şu', 'o', 'bir', 'ama', 'ancak', 
        'fakat', 'çünkü', 'için', 'gibi', 'kadar', 'daha', 'en', 'çok',
        'az', 'ne', 'nasıl', 'neden', 'niçin', 'nerede', 'kim', 'hangi',
        'mi', 'mu', 'mı', 'mü', 'da', 'de', 'ta', 'te', 'ki', 'ise'
    }
    
    sentence_scores = {}
    for idx, sentence in enumerate(sentences):
        score = 0
        words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
        for word in words_in_sentence:
            if word not in stopwords and len(word) > 3:
                score += word_freq.get(word, 0)
        
        # Cümle uzunluğunu normalize et
        if len(words_in_sentence) > 0:
            sentence_scores[idx] = score / len(words_in_sentence)
    
    # En yüksek skorlu cümleleri seç (orijinal sıralamayı koru)
    top_sentence_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentence_indices = sorted([idx for idx, score in top_sentence_indices])
    
    summary_sentences = [sentences[idx] for idx in top_sentence_indices]
    return ". ".join(summary_sentences) + "."

# 2. YAKE ile Anahtar Kelime Çıkarma
def extract_keywords_yake(text, top_n=15):
    """YAKE algoritması ile anahtar kelimeleri çıkar"""
    # YAKE parametreleri
    language = "tr"  # Türkçe
    max_ngram_size = 2  # Tek kelime ve iki kelimeli ifadeler
    deduplication_threshold = 0.9
    num_of_keywords = top_n
    
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        top=num_of_keywords,
        features=None
    )
    
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords

# 3. Basit TF-IDF benzeri yöntem (yedek)
def extract_keywords_simple(text, top_n=10):
    """Basit frekans bazlı anahtar kelime çıkarma"""
    turkish_stopwords = {
        've', 'veya', 'ile', 'bu', 'şu', 'o', 'bir', 'ama', 'ancak', 
        'fakat', 'çünkü', 'için', 'gibi', 'kadar', 'daha', 'en', 'çok',
        'az', 'ne', 'nasıl', 'neden', 'niçin', 'nerede', 'kim', 'hangi',
        'mi', 'mu', 'mı', 'mü', 'da', 'de', 'ta', 'te', 'ya', 'ki',
        'ise', 'eğer', 'olarak', 'üzere', 'dolayı', 'hem', 'yani'
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if w not in turkish_stopwords and len(w) > 3]
    word_freq = Counter(words)
    return word_freq.most_common(top_n)

# Özet oluştur
print("\n📝 ÖZET (2 en önemli cümle):")
summary = advanced_extract_summary(full_text, num_sentences=2)
print(f"{summary}")

# YAKE ile anahtar kelimeleri çıkar
print(f"\n🔑 ANAHTAR KELİMELER (YAKE Algoritması):")
try:
    keywords_yake = extract_keywords_yake(full_text, top_n=15)
    for idx, (keyword, score) in enumerate(keywords_yake, 1):
        print(f"  {idx}. {keyword} (skor: {score:.4f})")
except Exception as e:
    print(f"  YAKE hatası: {e}")
    print("\n🔑 ANAHTAR KELİMELER (Basit Yöntem):")
    keywords_simple = extract_keywords_simple(full_text, top_n=10)
    for idx, (word, freq) in enumerate(keywords_simple, 1):
        print(f"  {idx}. {word} ({freq} kez)")

# Metin İstatistikleri
word_count = len(full_text.split())
char_count = len(full_text)
sentence_count = len(re.split(r'[.!?]+', full_text))
unique_words = len(set(re.findall(r'\b\w+\b', full_text.lower())))

print(f"\n📊 DETAYLI İSTATİSTİKLER:")
print(f"  • Toplam kelime sayısı: {word_count}")
print(f"  • Benzersiz kelime sayısı: {unique_words}")
print(f"  • Kelime çeşitliliği: {unique_words/word_count*100:.1f}%")
print(f"  • Toplam karakter sayısı: {char_count}")
print(f"  • Cümle sayısı: {sentence_count}")
print(f"  • Ortalama kelime/cümle: {word_count/max(sentence_count, 1):.1f}")
print(f"  • Segment sayısı: {len(all_transcriptions)}")
print(f"  • Ses süresi: {total_duration:.2f} saniye ({total_duration/60:.1f} dakika)")

# Sonuçları dosyaya kaydet
output_file = audio_path.replace('.wav', '_analysis.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("TRANSKRİPSİYON ANALİZİ\n")
    f.write("="*70 + "\n\n")
    
    f.write("ÖZET:\n")
    f.write(summary + "\n\n")
    
    f.write("ANAHTAR KELİMELER:\n")
    try:
        for idx, (keyword, score) in enumerate(keywords_yake, 1):
            f.write(f"  {idx}. {keyword}\n")
    except:
        for idx, (word, freq) in enumerate(keywords_simple, 1):
            f.write(f"  {idx}. {word}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("TAM TRANSKRİPSİYON:\n")
    f.write("="*70 + "\n")
    f.write(full_text)

print(f"\n💾 Analiz raporu kaydedildi: {output_file}")
