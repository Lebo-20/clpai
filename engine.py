import os
import asyncio
import logging
import shutil
import subprocess
import datetime
import json
import re
from scenedetect import detect, ContentDetector, AdaptiveDetector, split_video_ffmpeg
import yt_dlp
import google.generativeai as genai
import cv2
import numpy as np
import whisper

logger = logging.getLogger(__name__)

class VideoEngine:
    def __init__(self, temp_dir="./temp"):
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        # Load whisper model lazily
        self._whisper_model = None
        
        # Configure Gemini
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            model_name = os.getenv("WHISPER_MODEL", "base")
            device = os.getenv("WHISPER_DEVICE", "cpu")
            logger.info(f"Loading Whisper model '{model_name}' on {device}...")
            self._whisper_model = whisper.load_model(model_name, device=device)
        return self._whisper_model

    async def analyze_download_error_with_ai(self, error_log: str):
        """Uses Gemini to analyze yt-dlp errors and provide a technical fix."""
        if not self.api_key:
            return None
            
        prompt = f"""
        Kamu adalah AI engineer khusus memperbaiki error yt-dlp.
        Analisa log di bawah, tentukan penyebab, dan berikan fix.
        
        Output WAJIB JSON:
        {{
          "error_type": "format_not_available / login_required / region_block / unknown",
          "reason": "penjelasan singkat",
          "fix": {{
            "format": "format string",
            "cookies": true/false,
            "additional_flags": ["flag1"]
          }}
        }}

        Error log:
        {error_log}
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = await asyncio.to_thread(model.generate_content, prompt)
            json_match = re.search(r'{{.*}}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception:
            return None

    async def get_video_info(self, url: str):
        """Extracts available formats and subtitles for a given URL."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cookies = os.path.join(base_dir, "cookies.txt")
        
        opts = {
            'quiet': True,
            'no_warnings': True,
            'cookiefile': cookies if os.path.exists(cookies) else None,
            'socket_timeout': 30, # Increase timeout to 30s
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web'],
                }
            }
        }
        
        def _extract():
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
                
        try:
            info = await asyncio.to_thread(_extract)
            
            # Formats
            formats = []
            seen_heights = set()
            for f in sorted(info.get('formats', []), key=lambda x: (x.get('height', 0) or 0), reverse=True):
                h = f.get('height')
                if h and h >= 360 and h <= 1080 and h not in seen_heights:
                    formats.append({'id': f.get('format_id'), 'height': h, 'ext': f.get('ext')})
                    seen_heights.add(h)
            
            # Subtitles
            subtitles = {}
            if 'subtitles' in info:
                for lang, data in info['subtitles'].items():
                    subtitles[lang] = lang
            if 'automatic_captions' in info:
                for lang, data in info['automatic_captions'].items():
                    if lang not in subtitles:
                        subtitles[lang] = f"{lang} (auto)"
                        
            return {
                'title': info.get('title'),
                'formats': formats[:6],
                'subtitles': subtitles
            }
        except Exception as e:
            logger.error(f"Error fetching video info: {e}")
            return None

    def get_video_duration(self, video_path):
        """Returns video duration in seconds using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return float(result.stdout.strip())
        except:
            return 0

    def validate_video(self, video_path):
        """Checks if video exists, is not empty, and has duration."""
        if not os.path.exists(video_path):
            raise Exception("File video tidak ditemukan setelah download.")
        
        size_kb = os.path.getsize(video_path) / 1024
        if size_kb < 10: # Minimum 10KB
            raise Exception(f"File video corrupt atau kosong (Size: {size_kb:.1f} KB).")
        
        duration = self.get_video_duration(video_path)
        if duration <= 0:
            raise Exception("Video tidak memiliki durasi (Corrupt file atau format tidak didukung).")
            
        return duration

    async def download_video(self, url: str, job_id: str, options: dict = {}, progress_callback=None) -> str:
        """Downloads video with AI-powered self-healing and specific format fallback logic."""
        output_template = os.path.join(self.temp_dir, f"{job_id}_input.%(ext)s")
        
        target_format = options.get('format_id')
        target_lang = options.get('subtitle_lang')
        
        def yt_dlp_hook(d):
            if d['status'] == 'downloading':
                try:
                    p_str = d.get('_percent_str', '0%').replace('%', '').strip()
                    p_float = float(p_str)
                    # Map 0-100% download to 5-30% job progress
                    job_p = 5 + (p_float / 100 * 25)
                    if progress_callback:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(progress_callback(job_p, f"Downloading... ({p_float:.1f}%)"), loop)
                except Exception:
                    pass

        # Sequence of formats
        format_fallbacks = [
            "(bv*+ba/b/bv+ba/best)",
            'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'best',
            None
        ]
        
        # If user picked a specific format, we put it first
        if target_format:
            format_fallbacks.insert(0, f"{target_format}+bestaudio/best")

        current_opts = {
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'noplaylist': True,
            'retries': 15,
            'socket_timeout': 60, # 60 seconds for actual download
            'progress_hooks': [yt_dlp_hook],
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
            }
        }
        
        if target_lang:
            current_opts['writesubtitles'] = True
            current_opts['subtitleslangs'] = [target_lang]

        # Load site-specific cookies
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cookie_file = None

        # Comprehensive site-specific cookie mapping
        site_cookie_map = {
            "youtube.com": "cookies_youtube.txt",
            "youtu.be": "cookies_youtube.txt",
            "tiktok.com": "cookies_tiktok.txt",
            "facebook.com": "cookies_facebook.txt",
            "fb.watch": "cookies_facebook.txt",
            "instagram.com": "cookies_instagram.txt",
            "twitter.com": "cookies_twitter.txt",
            "x.com": "cookies_twitter.txt"
        }

        for domain, filename in site_cookie_map.items():
            if domain in url:
                path = os.path.join(base_dir, filename)
                if os.path.exists(path):
                    cookie_file = path
                    logger.info(f"Using site-specific cookies for {domain}: {filename}")
                    break
        
        # Site-specific optimized arguments
        if "youtube.com" in url or "youtu.be" in url:
            current_opts['extractor_args'] = {
                'youtube': {
                    'player_client': ['android', 'ios', 'web'],
                    'skip': ['dash', 'hls']
                }
            }
        
        # Fallback to general cookies.txt if no specific cookie file was set or found
        if not cookie_file:
            general_cookies = os.path.join(base_dir, "cookies.txt")
            if os.path.exists(general_cookies):
                cookie_file = general_cookies
                logger.info(f"Using general cookies.txt fallback")
        
        if cookie_file:
            current_opts['cookiefile'] = cookie_file
        else:
            logger.warning("No cookie file found! Download might fail for restricted videos.")

        max_attempts = len(format_fallbacks)
        for attempt in range(max_attempts):
            f_str = format_fallbacks[attempt]
            if f_str:
                current_opts['format'] = f_str
            else:
                current_opts.pop('format', None) # Default yt-dlp
            
            try:
                logger.info(f"Download attempt {attempt+1} using format: {f_str or 'default'}")
                def _download():
                    with yt_dlp.YoutubeDL(current_opts) as ydl:
                        # Pre-check if video is available
                        info = ydl.extract_info(url, download=True)
                        return ydl.prepare_filename(info)
                path = await asyncio.to_thread(_download)
                # Final validation
                self.validate_video(path)
                return path

            except Exception as e:
                err_log = str(e)
                logger.warning(f"Attempt {attempt+1} failed: {err_log[:80]}...")
                
                # Check for explicit authentication/restriction errors
                auth_errors = [
                    "Sign in to confirm your age",
                    "This video is private",
                    "confirm you're not a bot"
                ]
                if any(x.lower() in err_log.lower() for x in auth_errors):
                    error_msg = f"❌ Akses Ditolak: Video ini butuh login valid. "
                    if "age" in err_log.lower():
                        error_msg += "Video ini dibatasi umur (Age Restricted)."
                    elif "private" in err_log.lower():
                        error_msg += "Video ini bersifat Private."
                    else:
                        error_msg += "Terdeteksi verifikasi bot."
                    
                    error_msg += "\n\nSilakan update `cookies.txt` via Telegram untuk melanjutkan."
                    raise Exception(error_msg)

                # If it's a format error, continue to next fallback
                if any(x in err_log for x in ["Requested format is not available", "format is not available"]):
                    if attempt < max_attempts - 1:
                        continue
                
                # AI search for fix (Only for non-auth errors)
                if attempt < max_attempts - 1:
                    solution = await self.analyze_download_error_with_ai(err_log)
                    if solution and "fix" in solution:
                        fix = solution["fix"]
                        if fix.get("format"): 
                            current_opts['format'] = fix['format']
                        if fix.get("additional_flags"):
                            # Map some common flags to opts if possible
                            for flag in fix["additional_flags"]:
                                if "--geo-bypass" in flag: current_opts['geo_bypass'] = True
                                if "--user-agent" in flag: 
                                    ua_match = re.search(r'--user-agent\s+"?([^"]+)"?', flag)
                                    if ua_match: current_opts['headers']['User-Agent'] = ua_match.group(1)
                                if "--extractor-args" in flag:
                                    ext_match = re.search(r'--extractor-args\s+"?([^"]+)"?', flag)
                                    if ext_match:
                                        logger.info(f"AI suggested extractor args: {ext_match.group(1)}")

                        logger.info(f"AI suggested fix: {solution.get('reason')}")
                        await asyncio.sleep(2)
                        continue
                
                if attempt == max_attempts - 1:
                    error_msg = f"Gagal mendownload video setelah {max_attempts} percobaan. "
                    if "cookies" in err_log.lower() or "login" in err_log.lower() or "sign in" in err_log.lower():
                        error_msg += "Video ini butuh login. Update cookies.txt via Telegram."
                    else:
                        error_msg += "Kemungkinan video diblokir, private, atau butuh cookies terbaru."
                    raise Exception(error_msg)

    async def analyze_with_gemini(self, transcript_segments: list):
        """Uses Gemini AI to find the most viral segments based on transcript with scoring."""
        if not self.api_key:
            return None

        text_content = ""
        for seg in transcript_segments:
            text_content += f"[{seg['start']:.1f} - {seg['end']:.1f}] {seg['text']}\n"

        prompt = f"""
        Pilih top 15 bagian paling menarik (skor 1-10).
        Skor tinggi (+3) untuk emosi kuat, (+2) kata kunci unik (anjir, ternyata, serem), (+2) plot twist.
        Format JSON list: [{{"title": "clickbait", "start": 0, "end": 0, "score": 10}}]
        Transkrip:
        {text_content}
        """

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = await asyncio.to_thread(model.generate_content, prompt)
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                return sorted(json.loads(json_match.group()), key=lambda x: x.get('score', 0), reverse=True)
            return None
        except Exception:
            return None

    def _get_smart_crop_params(self, video_path, target_ratio=9/16):
        """Returns the ffmpeg crop string centered on detected faces."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "ih*9/16:ih" # Fallback

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        target_w = int(h * target_ratio)
        if target_w >= w:
            cap.release()
            return f"{w}:{h}" # No crop needed
            
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        x_centers = []
        
        for pos in [0.3, 0.5, 0.7]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pos))
            ret, frame = cap.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Sort by size to get the biggest face (likely the speaker)
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, fw, fh = faces[0]
                x_centers.append(x + fw/2)

        cap.release()
        
        if not x_centers:
            # Fallback to center
            return f"ih*9/16:ih:(in_w-out_w)/2:0"
            
        avg_x = sum(x_centers) / len(x_centers)
        x_offset = int(avg_x - (target_w / 2))
        x_offset = max(0, min(x_offset, w - target_w))
        
        return f"{target_w}:{h}:{x_offset}:0"

    async def create_clips(self, input_path: str, job_id: str, options: dict, progress_callback=None):
        """Processes video into clips with granular progress reporting."""
        output_folder = os.path.join(self.temp_dir, f"{job_id}_clips")
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        mode = options.get("mode", "auto")
        do_subtitles = options.get("subtitles", False)
        do_vertical = options.get("vertical", True)
        target_duration = options.get("duration", 60)
        
        # PRE-VALIDATION before heavy processing
        video_duration = await asyncio.to_thread(self.validate_video, input_path)
        
        clips_to_process = []

        # STAGE 2: Transcribe (30-50%)
        if progress_callback: await progress_callback(30, "Transcribing audio...")
        
        result = None
        try:
            # Whisper will crash if audio is empty or missing
            # Disable fp16 on CPU to avoid warnings and potential issues
            device = os.getenv("WHISPER_DEVICE", "cpu")
            use_fp16 = (device == "cuda")
            result = await asyncio.to_thread(self.whisper_model.transcribe, input_path, fp16=use_fp16)
        except Exception as e:
            logger.warning(f"Whisper failed (likely no audio): {e}")
            result = {"segments": []} # Fallback to empty segments
        
        # STAGE 3: Analysis/Scene Detection (50-65%)
        if progress_callback: await progress_callback(50, "Analyzing scenes & scoring...")
        
        if mode == "ai" and self.api_key:
            ai_segments = await self.analyze_with_gemini(result["segments"])
            if ai_segments:
                for seg in ai_segments[:15]:
                    # Respect maximum duration from settings and video limit
                    start = min(seg['start'], video_duration)
                    end = min(seg['end'], start + target_duration, video_duration)
                    clips_to_process.append(((start, end), seg['title']))

        if not clips_to_process:
            # If Auto, we either use scene detect or fixed segments
            if mode == "auto":
                # Distribute clips across the entire video duration
                max_clips = 15
                # Calculate interval to spread clips evenly
                step = video_duration / max_clips
                
                for i in range(max_clips):
                    start = i * step
                    # Make sure the clip doesn't go over the total duration
                    end = min(start + target_duration, video_duration)
                    
                    if end - start > 2: # Ignore clips too short
                        clips_to_process.append(((start, end), f"clip_{i+1}"))
                    
                    if end >= video_duration:
                        break
            else:
                # Scene detection fallback
                scene_list = detect(input_path, AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=30))
                for i, scene in enumerate(scene_list[:15]):
                    start = min(scene[0].get_seconds(), video_duration)
                    end = min(scene[1].get_seconds(), start + target_duration, video_duration)
                    clips_to_process.append(((start, end), f"clip_{i+1}"))

        # STAGE 4: Rendering (65-95%)
        final_files = []
        total_clips = len(clips_to_process)
        
        for i, (times, title) in enumerate(clips_to_process):
            # Detailed progress: 65% + (i * step)
            step = 30 / total_clips
            p = 65 + (i * step)
            if progress_callback: await progress_callback(round(p, 1), f"Rendering clip {i+1}/{total_clips}...")
            
            start_t, end_t = times
            clip_name = f"clip_{i+1:02d}.mp4"
            temp_path = os.path.join(output_folder, f"raw_{clip_name}")
            final_path = os.path.join(output_folder, clip_name)

            # FFmpeg execution
            cmd = ['ffmpeg', '-y', '-ss', str(start_t), '-to', str(end_t), '-i', input_path, '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', '-c:a', 'aac', temp_path]
            subprocess.run(cmd, check=True, capture_output=True)
            
            curr = temp_path
            if do_vertical:
                v_path = os.path.join(output_folder, f"v_{clip_name}")
                crop_params = await asyncio.to_thread(self._get_smart_crop_params, curr)
                subprocess.run(['ffmpeg', '-y', '-i', curr, '-vf', f"crop={crop_params}", '-c:a', 'copy', v_path], check=True, capture_output=True)
                curr = v_path

            if do_subtitles:
                s_path = os.path.join(output_folder, f"sub_{clip_name}")
                srt_path = os.path.join(output_folder, f"sub_{i}.srt")
                
                # REUSE global transcribe result (much faster!)
                clip_segments = []
                for seg in result["segments"]:
                    if seg['start'] >= start_t and seg['end'] <= end_t:
                        # Normalize timestamps relative to clip start
                        new_seg = seg.copy()
                        new_seg['start'] -= start_t
                        new_seg['end'] -= start_t
                        clip_segments.append(new_seg)
                
                if clip_segments:
                    self._write_srt(clip_segments, srt_path)
                    escaped = srt_path.replace("\\", "/").replace(":", "\\:")
                    
                    # Custom styling based on orientation (Vertical vs Horizontal)
                    if do_vertical:
                        # Style for Vertical: Standard Symbols PS, Size 10, Bold, Outline 1, MarginV 90
                        style = "Fontname=Standard Symbols PS,PrimaryColour=&HFFFFFF,SecondaryColour=&H000000,OutlineColour=&H000000,BorderStyle=1,Outline=1,Shadow=0,Alignment=2,FontSize=10,Bold=1,MarginV=90"
                    else:
                        # Style for Horizontal: Nimbus Sans Narrow, Size 24, MarginV 8
                        style = "Fontname=Nimbus Sans Narrow,PrimaryColour=&HFFFFFF,SecondaryColour=&H000000,OutlineColour=&H000000,BorderStyle=1,Outline=1,Shadow=0,Alignment=2,FontSize=24,Bold=0,MarginV=8"
                    
                    cmd = ['ffmpeg', '-y', '-i', curr, '-vf', f"subtitles='{escaped}':force_style='{style}'", '-c:a', 'copy', s_path]
                    subprocess.run(cmd, check=True, capture_output=True)
                    curr = s_path
                else:
                    logger.info(f"No speech detected for clip {i+1}, skipping subtitles.")

            shutil.move(curr, final_path)
            final_files.append(final_path)

        # STAGE 5: Merging (95-100%)
        if progress_callback: await progress_callback(95, "Finalizing highlight video...")
        if len(final_files) > 1:
            m_path = os.path.join(output_folder, "final_highlight.mp4")
            l_path = os.path.join(output_folder, "list.txt")
            with open(l_path, "w") as f:
                for fp in final_files: f.write(f"file '{os.path.basename(fp)}'\n")
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', l_path, '-c', 'copy', m_path], check=True, capture_output=True)
            final_files.append(m_path)

        return final_files

    def _write_srt(self, segments, path):
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                f.write(f"{i+1}\n{self._format_timestamp(seg['start'])} --> {self._format_timestamp(seg['end'])}\n{seg['text'].strip()}\n\n")

    def _format_timestamp(self, seconds):
        td = datetime.timedelta(seconds=seconds)
        ts = str(td)
        if "." in ts:
            return ts.replace(".", ",")[:-3]
        return ts + ",000"

    async def cleanup(self, *paths):
        for path in paths:
            try:
                if not path: continue
                if os.path.isfile(path): os.remove(path)
                elif os.path.isdir(path): shutil.rmtree(path)
            except Exception as e: logger.error(f"Cleanup error: {e}")
