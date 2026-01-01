import gradio as gr
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import soundfile as sf
import yake
import re
from collections import Counter
import numpy as np

# Model yükleme (global olarak bir kez yüklenecek)
print("Model yükleniyor...")
model_dir = "./checkpoint-5"
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
print("Model hazır!")

def advanced_extract_summary(text, num_sentences=2):
    """Cümleleri skorlayarak en önemli cümleleri seç"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    if len(sentences) <= num_sentences:
        return ". ".join(sentences) + "."
    
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    
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
        
        if len(words_in_sentence) > 0:
            sentence_scores[idx] = score / len(words_in_sentence)
    
    top_sentence_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentence_indices = sorted([idx for idx, score in top_sentence_indices])
    
    summary_sentences = [sentences[idx] for idx in top_sentence_indices]
    return ". ".join(summary_sentences) + "."

def extract_keywords_yake(text, top_n=15):
    """YAKE algoritması ile anahtar kelimeleri çıkar"""
    try:
        custom_kw_extractor = yake.KeywordExtractor(
            lan="tr",
            n=2,
            dedupLim=0.9,
            top=top_n,
            features=None
        )
        keywords = custom_kw_extractor.extract_keywords(text)
        return keywords
    except:
        return []

def process_audio(audio_file, num_summary_sentences=2, num_keywords=10):
    """Ses dosyasını işle: transkripsiyon + NLP analizi"""
    
    if audio_file is None:
        return "❌ Lütfen bir ses dosyası yükleyin!", "", "", ""
    
    try:
        # Ses dosyasını oku
        audio_input, sample_rate = sf.read(audio_file)
        
        # Stereo ise mono'ya çevir
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)
        
        # Eğer örnekleme hızı 16000 değilse yeniden örnekle
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            from scipy.signal import resample
            num_samples = int(len(audio_input) * target_sample_rate / sample_rate)
            audio_input = resample(audio_input, num_samples)
            sample_rate = target_sample_rate
        
        # 30 saniyelik segmentlere böl ve transkribe et
        segment_duration = 30
        segment_samples = target_sample_rate * segment_duration
        total_samples = len(audio_input)
        total_duration = total_samples / target_sample_rate
        
        all_transcriptions = []
        progress_text = f"⏱ Toplam süre: {total_duration:.1f} saniye ({total_duration/60:.1f} dakika)\n\n"
        progress_text += " İşleniyor...\n"
        
        for i in range(0, total_samples, segment_samples):
            start = i
            end = min(i + segment_samples, total_samples)
            segment = audio_input[start:end]
            
            if len(segment) == 0:
                continue
            
            inputs = processor(segment, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            all_transcriptions.append(transcription)
        
        # Tam metin
        full_text = " ".join(all_transcriptions)
        
        if not full_text.strip():
            return "❌ Transkripsiyon oluşturulamadı!", "", "", ""
        
        # NLP Analizi
        summary = advanced_extract_summary(full_text, num_sentences=int(num_summary_sentences))
        
        # Anahtar kelimeler
        keywords_yake = extract_keywords_yake(full_text, top_n=int(num_keywords))
        keywords_text = ""
        if keywords_yake:
            for idx, (keyword, score) in enumerate(keywords_yake, 1):
                keywords_text += f"{idx}. {keyword} (skor: {score:.4f})\n"
        else:
            keywords_text = "Anahtar kelime çıkarılamadı."
        
        # İstatistikler
        word_count = len(full_text.split())
        char_count = len(full_text)
        sentence_count = len(re.split(r'[.!?]+', full_text))
        unique_words = len(set(re.findall(r'\b\w+\b', full_text.lower())))
        
        stats_text = f"""
📊 İSTATİSTİKLER:
• Toplam kelime sayısı: {word_count}
• Benzersiz kelime sayısı: {unique_words}
• Kelime çeşitliliği: {unique_words/word_count*100:.1f}%
• Cümle sayısı: {sentence_count}
• Ortalama kelime/cümle: {word_count/max(sentence_count, 1):.1f}
• Ses süresi: {total_duration:.1f} saniye ({total_duration/60:.1f} dakika)
"""
        
        return full_text, summary, keywords_text, stats_text
        
    except Exception as e:
        return f"❌ Hata oluştu: {str(e)}", "", "", ""

# Gradio Arayüzü
with gr.Blocks(title="📢 Ses Dosyası İceriginizi Öğrenin! ", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📢 Ses Dosyanızdan özet ve Analiz Çıkarın
    Ses dosyalarınızı yükleyin, transkripsiyon alın ve otomatik özet + anahtar kelimeler çıkarın!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎵 Ses Dosyası Yükleyin",
                type="filepath",
                sources=["upload"]
            )
            
            with gr.Row():
                summary_slider = gr.Slider(
                    minimum=1, 
                    maximum=5, 
                    value=2, 
                    step=1,
                    label="📝 Özet Cümle Sayısı"
                )
                keyword_slider = gr.Slider(
                    minimum=5, 
                    maximum=20, 
                    value=10, 
                    step=1,
                    label="🔑 Anahtar Kelime Sayısı"
                )
            
            process_btn = gr.Button("▶️ İşlemeyi Başlat", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            with gr.Tab("📄 Transkripsiyon"):
                transcription_output = gr.Textbox(
                    label="Tam Transkripsiyon",
                    lines=15,
                    max_lines=20
                )
            
            with gr.Tab("📝 Özet"):
                summary_output = gr.Textbox(
                    label="Otomatik Özet",
                    lines=5
                )
            
            with gr.Tab("🔑 Anahtar Kelimeler"):
                keywords_output = gr.Textbox(
                    label="Anahtar Kelimeler (YAKE)",
                    lines=12
                )
            
            with gr.Tab("📊 İstatistikler"):
                stats_output = gr.Textbox(
                    label="Metin İstatistikleri",
                    lines=10
                )
    
    # Örnek dosyalar
    gr.Markdown("### 📁 Örnek Dosyalar")
    gr.Examples(
        examples=[
            ["mennan1.wav"],
            ["mennan2.wav"],
            ["mennan3.wav"],
            ["podcast1.wav"]
        ],
        inputs=audio_input,
        label="Örnek ses dosyalarından birini seçin"
    )
    
    # İşleme butonu event
    process_btn.click(
        fn=process_audio,
        inputs=[audio_input, summary_slider, keyword_slider],
        outputs=[transcription_output, summary_output, keywords_output, stats_output]
    )
    
    gr.Markdown("""
    ---
    **Nasıl Kullanılır?**
    1. Ses dosyanızı yükleyin (WAV, MP3, vb.)
    2. Özet cümle sayısını ve anahtar kelime sayısını ayarlayın
    3. "İşlemeyi Başlat" butonuna tıklayın
    4. Sonuçları farklı sekmelerde inceleyin
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, inbrowser=True)
