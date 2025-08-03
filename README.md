# 🎬 Video Story Bot

This project automatically generates AI-powered storytelling videos by:
- Creating images from text prompts (via FusionBrain API)
- Generating audio with realistic TTS (Edge TTS)
- Syncing subtitles word-by-word (using Whisper)
- Compiling short videos and sending them via Telegram

# 🤝 Authors
**Sohaib Essam** — [GitHub Profile](https://github.com/Sohaib010) 

### Project structure
---

## 🔧 Features

✅ FusionBrain integration for image generation  
✅ Edge-TTS for realistic speech  
✅ Word-synced subtitles with Whisper  
✅ Video generation with MoviePy  
✅ Excel-based batch processing  
✅ Telegram video delivery

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/text2video-bot.git
cd text2video-bot


```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```
### 3. Prepare input data
Place your Excel file (e.g. data.xlsx) in the project root. It should contain the following columns:

- Image Sentences
- Audio Sentences
- Video Title
- Video Description
- Hashtags


### 4. Set up API credentials
In the bot.py (or your main file):

Replace the placeholders:

python
Копировать
Редактировать
API_KEY = "YOUR_FUSIONBRAIN_API_KEY"
SECRET_KEY = "YOUR_FUSIONBRAIN_SECRET"
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = ["YOUR_TELEGRAM_CHAT_ID"]

### 5. Run the script

### On Windows:
```bash
run.bat
```
### On Linux/macOS:
```bash
./run.sh
```
### Or manually:
```bash
python bot.py
```
### 💡 Ideas for improvement

✅ Subtitle overlay support

✅ Selectable TTS voices

✅ Web dashboard for uploading Excel files

✅ Preview mode before sending to Telegram

✅ YouTube Shorts export pipeline
