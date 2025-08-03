import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import asyncio
import edge_tts
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, VideoFileClip, CompositeVideoClip
import logging
import tempfile
import pandas as pd
import numpy as np
from datetime import timedelta
import pysrt
import subprocess
from io import BytesIO
import whisper
import re
import time
import os
import glob

# ======================
# CONFIGURATION SECTION
# ======================

# Telegram Config
CHAT_ID = ["YOUR_CHAT_ID_1", "YOUR_CHAT_ID_2"]  # ← Replace with your actual Telegram user/chat IDs
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"           # ← Replace with your bot token from BotFather

# API Keys
API_KEY = "YOUR_FUSIONBRAIN_API_KEY"            # ← Replace with your FusionBrain API key
SECRET_KEY = "YOUR_FUSIONBRAIN_SECRET_KEY"      # ← Replace with your FusionBrain secret key

# Video Configuration
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 24
SUBTITLE_POSITION = ('center', 1300)  # (x, y) position for subtitles
WORD_ANIMATION_DURATION = 0.2  # Duration of fade in/out animation for words

# ======================
# INITIALIZATION
# ======================

# Logging Setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Load Whisper model once
WHISPER_MODEL = whisper.load_model("small")

# ======================
# FUSIONBRAIN API CLASS
# ======================

class FusionBrainAPI:
    def __init__(self, base_url, api_key, secret_key):
        self.BASE_URL = base_url
        self.HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }
        logger.info("✅ FusionBrain API initialized")

    def get_pipeline(self):
        """Get available pipeline ID"""
        try:
            response = requests.get(
                f"{self.BASE_URL}key/api/v1/pipelines",
                headers=self.HEADERS
            )
            response.raise_for_status()
            pipeline_id = response.json()[0]['id']
            logger.info(f"🔧 Pipeline ID: {pipeline_id}")
            return pipeline_id
        except Exception as e:
            logger.error(f"❌ Failed to get pipeline: {e}")
            return None

    def generate_image(self, prompt, pipeline_id, width=1080, height=1920):
        """Generate image from text prompt"""
        params = {
            "type": "GENERATE",
            "numImages": 1,
            "width": width,
            "height": height,
            "generateParams": {"query": prompt}
        }

        data = {
            'pipeline_id': (None, pipeline_id),
            'params': (None, json.dumps(params), 'application/json')
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}key/api/v1/pipeline/run",
                headers=self.HEADERS,
                files=data
            )
            response.raise_for_status()
            request_id = response.json()['uuid']
            logger.info(f"🖼️ Generation started - Request ID: {request_id}")
            return request_id
        except Exception as e:
            logger.error(f"❌ Failed to start generation: {e}")
            return None

    def check_generation_status(self, request_id, max_attempts=20, delay=5):
        """Check status of image generation"""
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.BASE_URL}key/api/v1/pipeline/status/{request_id}",
                    headers=self.HEADERS
                )
                response.raise_for_status()
                data = response.json()

                if data['status'] == 'DONE':
                    logger.info("🎉 Image generation completed")
                    return data['result']['files'][0]  # Return first image
                elif data['status'] == 'FAILED':
                    logger.error(f"❌ Generation failed: {data.get('message', 'Unknown error')}")
                    return None

            except Exception as e:
                logger.error(f"⚠️ Status check failed (attempt {attempt + 1}): {e}")

            time.sleep(delay)

        logger.error("⌛ Max attempts reached - generation timeout")
        return None

# Initialize API
fusion_brain = FusionBrainAPI(
    base_url='https://api-key.fusionbrain.ai/',
    api_key=API_KEY,
    secret_key=SECRET_KEY
)

# ======================
# TEXT & IMAGE FUNCTIONS
# ======================

def create_text_image(text, video_size, padding=40, corner_radius=40, 
                     bg_color=(0, 0, 0, 250), text_color='yellow', font_size=60):
    """Create an image with text on a rounded rectangle background"""
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        img_size = (text_width + 2*padding, text_height + 2*padding)
        text_img = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)

        draw.rounded_rectangle(
            [0, 0, img_size[0], img_size[1]],
            radius=corner_radius,
            fill=bg_color
        )

        draw.text((padding, padding), text, font=font, fill=text_color)

        buffer = BytesIO()
        text_img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"❌ Failed to create text image: {e}")
        return None

# ======================
# VIDEO PROCESSING FUNCTIONS
# ======================

async def generate_image_for_text(text, pipeline_id):
    """Generate image from text using FusionBrain API"""
    request_id = fusion_brain.generate_image(text, pipeline_id)
    if not request_id:
        return None
        
    image_data = fusion_brain.check_generation_status(request_id)
    if not image_data:
        return None
        
    try:
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"❌ Failed to process generated image: {e}")
        return None

async def create_video_clip(image, audio_file, sentence, index):
    """Create video clip from image and audio with word-by-word sync"""
    try:
        # Save image to temp file
        img_path = f"temp_img_{index}.png"
        image.save(img_path)
        
        # Load audio
        audio_clip = AudioFileClip(audio_file)
        total_duration = audio_clip.duration
        
        # Get precise word timings
        word_timings = get_word_timings(audio_file)
        
        # Create clips for each word
        word_clips = []
        for word, start, end in word_timings:
          # Create word image
          buffer = create_text_image(word, image.size, font_size=120)
          if not buffer:
              continue
          
          # Convert buffer to numpy array
          img = Image.open(buffer)
          img_array = np.array(img)
          
          # Create clip for this word
          duration = end - start
          word_clip = ImageClip(img_array)
          word_clip = word_clip.set_duration(duration)
          word_clip = word_clip.set_start(start)
          word_clip = word_clip.set_position(SUBTITLE_POSITION)
          
          # Add fade animation
          #word_clip = word_clip.crossfadein(WORD_ANIMATION_DURATION).crossfadeout(WORD_ANIMATION_DURATION)
          word_clips.append(word_clip)
        
        # Create base video clip
        video_clip = ImageClip(img_path).set_duration(total_duration)
        video_clip = video_clip.set_audio(audio_clip)
        
        # Composite all clips together
        final_clip = CompositeVideoClip([video_clip] + word_clips)
        
        # Save to temp file
        output_path = f"temp_vid_{index}.mp4"
        final_clip.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            preset='fast',
            threads=2
        )
        
        # Verify file was created
        while not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            await asyncio.sleep(0.5)
        
        return output_path
    except Exception as e:
        logger.error(f"❌ Failed to create video clip: {e}")
        return None
    finally:
        # Cleanup image file
        if os.path.exists(img_path):
            os.remove(img_path)

def get_word_timings(audio_file):
    """Get precise word timings using Whisper"""
    try:
        result = WHISPER_MODEL.transcribe(audio_file, word_timestamps=True)
        timings = []
        for segment in result['segments']:
            for word in segment['words']:
                timings.append((
                    word['word'],
                    word['start'],
                    word['end']
                ))
        return timings
    except Exception as e:
        logger.error(f"❌ Failed to get word timings: {e}")
        return []

def concatenate_videos(video_paths, output_path):
    """Concatenate multiple video clips into one"""
    try:
        clips = [VideoFileClip(path) for path in video_paths]
        final_clip = concatenate_videoclips(clips, method="compose")
        
        final_clip.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            preset='fast',
            threads=4
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to concatenate videos: {e}")
        return False

def extract_audio(video_path, audio_path):
    """Extract audio from video file"""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-q:a', '0', '-map', 'a', audio_path
        ], check=True)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract audio: {e}")
        return False

def generate_subtitles(audio_path, srt_path):
    """Generate subtitles from audio using Whisper"""
    try:
        logger.info("🔍 Generating subtitles...")
        result = WHISPER_MODEL.transcribe(audio_path)
        subs = pysrt.SubRipFile()
        for i, segment in enumerate(result['segments'], 1):
            start_time = timedelta(seconds=segment['start'])
            end_time = timedelta(seconds=segment['end'])
            subs.append(pysrt.SubRipItem(
                index=i,
                start=pysrt.SubRipTime(
                    hours=start_time.seconds // 3600,
                    minutes=(start_time.seconds % 3600) // 60,
                    seconds=start_time.seconds % 60,
                    milliseconds=start_time.microseconds // 1000
                ),
                end=pysrt.SubRipTime(
                    hours=end_time.seconds // 3600,
                    minutes=(end_time.seconds % 3600) // 60,
                    seconds=end_time.seconds % 60,
                    milliseconds=end_time.microseconds // 1000
                ),
                text=segment['text'].strip()
            ))
        subs.save(srt_path)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to generate subtitles: {e}")
        return False

def burn_subtitles(video_path, srt_path, output_path):
    """Burn subtitles into video"""
    try:
        video = VideoFileClip(video_path)
        subs = pysrt.open(srt_path)
        
        text_clips = []
        for sub in subs:
            start = timedelta(
                hours=sub.start.hours,
                minutes=sub.start.minutes,
                seconds=sub.start.seconds,
                milliseconds=sub.start.milliseconds
            ).total_seconds()
            
            end = timedelta(
                hours=sub.end.hours,
                minutes=sub.end.minutes,
                seconds=sub.end.seconds,
                milliseconds=sub.end.milliseconds
            ).total_seconds()
            
            duration = end - start
            text_clip = create_subtitle_clip(sub.text, start, duration, video.size)
            if text_clip:
                text_clips.append(text_clip)
        
        final = CompositeVideoClip([video] + text_clips)
        final.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            preset='fast',
            threads=4,
            ffmpeg_params=['-max_muxing_queue_size', '1024']
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to burn subtitles: {e}")
        return False

def create_subtitle_clip(text, start_time, duration, video_size):
    """Create a subtitle clip for video"""
    try:
        # Create text image
        text_buffer = create_text_image(text, video_size, font_size=80)
        if not text_buffer:
            return None
            
        # Convert to numpy array
        text_img = Image.open(text_buffer)
        text_array = np.array(text_img)
        
        # Create and configure clip
        clip = ImageClip(text_array)
        clip = clip.set_duration(duration)
        clip = clip.set_start(start_time)
        clip = clip.set_position(SUBTITLE_POSITION)
        
        return clip
    except Exception as e:
        logger.error(f"❌ Failed to create subtitle clip: {e}")
        return None

# ======================
# TELEGRAM FUNCTIONS
# ======================

def send_video_to_telegram(video_path, title, description, hashtags, row_num):
    """Send video to Telegram with metadata"""
    try:
        with open(video_path, 'rb') as video_file:
            for chat_id in CHAT_ID:
                # Send video
                video_file.seek(0)
                response = requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
                    files={'video': video_file},
                    data={'chat_id': chat_id, 'caption': f"{row_num}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"✘ Failed to send video to {chat_id}: {response.text}")
                else:
                    logger.info(f"✓ Video sent to {chat_id}")
                
                # Send metadata as separate messages
                if title:
                    send_text_message(f"Title: {title}", chat_id)
                if description:
                    send_text_message(f"Description: {description}", chat_id)
                if hashtags:
                    send_text_message(f"Tags: {hashtags}", chat_id)
                    
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send to Telegram: {e}")
        return False

def send_text_message(text, chat_id):
    """Send text message to Telegram"""
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                'chat_id': chat_id,
                'text': text
            }
        )
        if response.status_code != 200:
            logger.error(f"✘ Failed to send text: {response.text}")
    except Exception as e:
        logger.error(f"❌ Failed to send text message: {e}")

# ======================
# DATA PROCESSING FUNCTIONS
# ======================

def read_excel_data(file_path):
    """Read and process Excel data"""
    try:
        df = pd.read_excel(file_path)
        required_cols = ['Image Sentences', 'Audio Sentences', 
                        'Video Title', 'Video Description', 'Hashtags']
        
        if not all(col in df.columns for col in required_cols):
            logger.error(f"❌ Missing required columns: {required_cols}")
            return None

        rows_data = []
        for _, row in df.iterrows():
            # Get data from row
            img_text = row['Audio Sentences']
            audio_text = row['Image Sentences']
            title = row.get('Video Title', '')
            desc = row.get('Video Description', '')
            tags = row.get('Hashtags', '')

            # Split into sentences
            img_sentences = [s.strip() for s in re.split(r'[.]', img_text) if s.strip()]
            audio_sentences = [s.strip() for s in re.split(r'[.!?]', audio_text) if s.strip()]

            if len(img_sentences) != len(audio_sentences):
                logger.warning(f"⚠️ Sentence count mismatch: {len(img_sentences)} vs {len(audio_sentences)}")
                continue

            rows_data.append((img_sentences, audio_sentences, title, desc, tags))

        logger.info(f"📊 Processed {len(rows_data)} rows from Excel")
        return rows_data
        
    except Exception as e:
        logger.error(f"❌ Failed to read Excel file: {e}")
        return None

def cleanup_temp_files():
    """Remove temporary files"""
    patterns = ["temp_*.mp4", "temp_*.png", "temp_*.mp3"]
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass

# ======================
# MAIN EXECUTION
# ======================

async def process_row(row_data, row_index, pipeline_id):
    """Process a single row from Excel"""
    img_sentences, audio_sentences, title, desc, tags = row_data
    video_clips = []
    
    for idx, (img_sentence, audio_sentence) in enumerate(zip(img_sentences, audio_sentences)):
        logger.info(f"  🖼️ Processing sentence {idx + 1}/{len(img_sentences)}")
        
        # Generate image
        image = await generate_image_for_text(img_sentence, pipeline_id)
        if not image:
            continue
        
        # Generate speech
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            await edge_tts.Communicate(audio_sentence, "en-US-EmmaNeural").save(audio_file.name)
            
            # Create video clip
            clip_path = await create_video_clip(image, audio_file.name, audio_sentence, idx)
            if clip_path:
                video_clips.append(clip_path)
    
    if not video_clips:
        logger.warning(f"⚠️ No valid clips for row {row_index + 1}")
        return False
    
    # Concatenate clips
    final_video = f"final_row_{row_index + 1}.mp4"
    if not concatenate_videos(video_clips, final_video):
        return False
    
    # Add subtitles (optional, commented out in your code)
    # temp_audio = f"audio_row_{row_index + 1}.mp3"
    # if not extract_audio(final_video, temp_audio):
    #     return False
    # subtitles = f"subs_row_{row_index + 1}.srt"
    # if not generate_subtitles(temp_audio, subtitles):
    #     return False
    # final_output = f"final_with_subs_row_{row_index + 1}.mp4"
    # if not burn_subtitles(final_video, subtitles, final_output):
    #     return False
    
    # Send to Telegram
    if not send_video_to_telegram(
        final_video,  # Pass the video file path
        title,
        desc,
        tags,
        row_index + 1  # Pass the row number
    ):
        return False
    
    return True

async def main():
    """Main execution function"""
    try:
        # Read Excel data
        data = read_excel_data("data3.xlsx")
        if not data:
            return
            
        # Get pipeline ID
        pipeline_id = fusion_brain.get_pipeline()
        if not pipeline_id:
            return
            
        # Process each row
        for i, row in enumerate(data):
            logger.info(f"🔷 Processing row {i + 1}/{len(data)}")
            await process_row(row, i, pipeline_id)
            
    except Exception as e:
        logger.error(f"❌ Critical error in main: {e}")
    finally:
        cleanup_temp_files()
        logger.info("🏁 Processing complete")

if __name__ == '__main__':
    asyncio.run(main())



