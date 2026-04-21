import os
import asyncio
import logging
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile, Message, ReplyKeyboardMarkup, KeyboardButton, CallbackQuery
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

from engine import VideoEngine

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
MAX_PARALLEL_JOBS = int(os.getenv("MAX_PARALLEL_JOBS", 2))
TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
ADMIN_IDS = [int(i.strip()) for i in os.getenv("ADMIN_IDS", "").split(",") if i.strip()]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClipperBot")

# Initialize Bot and Dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
engine = VideoEngine(temp_dir=TEMP_DIR)

# Job Queue
queue = asyncio.Queue()

# User Settings Store (In-memory for now, could be DB)
user_settings = {}

def get_user_settings(user_id: int):
    if user_id not in user_settings:
        user_settings[user_id] = {
            "duration": 30,
            "mode": "ai", # Default to AI for best accuracy as requested
            "subtitles": False,
            "vertical": True,
            "max_clips": 5
        }
    return user_settings[user_id]

def is_admin(user_id: int) -> bool:
    return not ADMIN_IDS or user_id in ADMIN_IDS

def format_duration(seconds):
    if seconds <= 0:
        return "0s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def get_progress_bar(percentage, length=15):
    filled_len = int(length * percentage / 100)
    bar = "█" * filled_len + "░" * (length - filled_len)
    return f"`{bar}` {percentage:.1f}%"

async def worker():
    """Worker to process jobs from the queue."""
    while True:
        job = await queue.get()
        user_id = job['user_id']
        chat_id = job['chat_id']
        input_type = job['type']
        input_data = job['data']
        options = job['options']
        job_id = str(uuid.uuid4())[:8]
        job_start_time = datetime.now()
        
        # Initial status message
        status_msg = await bot.send_message(chat_id, f"📡 **Job `{job_id}`**: Menghubungkan ke server...")

        current_progress = {"p": 5, "stage": "Menghubungkan..."}
        job_active = True
        last_update_time = [0]
        update_lock = asyncio.Lock()

        async def update_progress(percentage, stage_msg, force=False):
            current_progress["p"] = percentage
            current_progress["stage"] = stage_msg
            
            now = datetime.now()
            elapsed = (now - job_start_time).total_seconds()
            
            # Rate limit: max once every 2.5 seconds, unless forced
            now_ts = now.timestamp()
            if not force and now_ts - last_update_time[0] < 2.5:
                return
            
            async with update_lock:
                last_update_time[0] = now_ts
                eta_str = "Calculating..."
                if percentage > 0.5: # Only show ETA after some progress
                    total_est = elapsed / (percentage / 100)
                    remaining = total_est - elapsed
                    eta_str = format_duration(remaining)
                
                progress_text = (
                    f"🚀 **Processing Job**: `{job_id}`\n"
                    f"📊 Stage: `{stage_msg}`\n\n"
                    f"{get_progress_bar(percentage)}\n"
                    f"⏱ Elapsed: {format_duration(elapsed)}\n"
                    f"⏳ ETA: {eta_str}"
                )
                try:
                    await bot.edit_message_text(text=progress_text, chat_id=chat_id, message_id=status_msg.message_id, parse_mode="Markdown")
                except Exception:
                    pass

        # Background Ticker to make the timer "alive"
        async def ticker():
            try:
                while job_active:
                    await asyncio.sleep(4) # Update every 4 seconds
                    if job_active:
                        await update_progress(current_progress["p"], current_progress["stage"])
            except asyncio.CancelledError:
                # Normal shutdown
                pass
            except Exception:
                pass

        ticker_task = asyncio.create_task(ticker())
        input_path = None
        try:
            logger.info(f"Processing job {job_id} for user {user_id}")
            
            if input_type == "full_download":
                # STAGE 1: Full Download
                await update_progress(5, "Downloading full video...")
                input_path = await engine.download_video(input_data, job_id, options=options, progress_callback=update_progress)
                
                # STAGE 2: Direct Upload
                job_active = False # Stop ticker
                ticker_task.cancel()
                
                await update_progress(95, "Uploading full video...")
                video = FSInputFile(input_path)
                await bot.send_video(chat_id, video, caption=f"✅ **Download Selesai!**\n\nID: `{job_id}`", parse_mode="Markdown")
                await update_progress(100, "Done!")
            else:
                # STAGE 1: Download (0-30%)
                await update_progress(5, "Downloading video...")
                if input_type == "url":
                    input_path = await engine.download_video(input_data, job_id, options=options, progress_callback=update_progress)
                else:
                    input_path = input_data
                await update_progress(30, "Download complete.")

                # STAGE 2-5: Processing (30-95%)
                clips = await engine.create_clips(input_path, job_id, options, progress_callback=update_progress)
                
                # Stop ticker before uploading
                job_active = False
                ticker_task.cancel()

            # STAGE 6: Upload (95-100%)
            if clips:
                await update_progress(96, "Uploading results...")
                # clips is now List[Tuple[path, title]]
                for i, (clip_path, title) in enumerate(clips):
                    is_merged = os.path.basename(clip_path) == "final_highlight.mp4"
                    video = FSInputFile(clip_path)
                    
                    if is_merged:
                        caption = f"🎬 **{title}**\n\nKompilasi momen terbaik pilihan AI."
                    else:
                        caption = f"🎬 **{title}**\n\nClip {i+1} dari {len(clips)-1 if len(clips)>1 else 1}"
                    
                    await bot.send_video(chat_id, video, caption=caption, parse_mode="Markdown")
                    await asyncio.sleep(1)
                
                await update_progress(100, "Done! Semua klip telah dikirim.")
            else:
                await update_progress(100, "Gagal: Tidak ada klip ditemukan.")

        except Exception as e:
            logger.error(f"Error in worker for job {job_id}: {e}")
            await bot.edit_message_text(text=f"❌ Job `{job_id}` gagal: {str(e)[:100]}", chat_id=chat_id, message_id=status_msg.message_id)
        finally:
            # Matikan ticker secara aman
            job_active = False
            if 'ticker_task' in locals():
                ticker_task.cancel()
                try:
                    await asyncio.wait_for(ticker_task, timeout=1.0)
                except:
                    pass
                
            if input_path:
                await engine.cleanup(input_path)
            if 'clips' in locals() and clips:
                await engine.cleanup(os.path.dirname(clips[0][0]))
                
            # FINAL CLEANUP: Delete status message and user trigger messages
            try:
                await bot.delete_message(chat_id, status_msg.message_id)
                last_menu_msgs.pop(user_id, None)
            except: pass
            
            trigger_ids = job.get('trigger_msg_ids', [])
            await delete_user_messages(chat_id, trigger_ids)
            
            queue.task_done()
            logger.info(f"Finished job {job_id}")

# Store last menu message ID to clean up
last_menu_msgs = {}

async def show_ui(chat_id: int, user_id: int, text: str, reply_markup=None, parse_mode="Markdown"):
    """Edits the existing bot message or sends a new one if it doesn't exist."""
    msg_id = last_menu_msgs.get(user_id)
    
    if msg_id:
        try:
            new_msg = await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode
            )
            return new_msg
        except Exception as e:
            # If edit fails (e.g. content same, or too old), we try to send a new one
            # and delete the old one to avoid stacking
            try:
                await bot.delete_message(chat_id, msg_id)
            except: pass
            
    try:
        new_msg = await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode
        )
        last_menu_msgs[user_id] = new_msg.message_id
        return new_msg
    except Exception as e:
        logger.error(f"UI Error: {e}")
        # Fallback to simple send if Markdown fails
        if parse_mode:
            new_msg = await bot.send_message(chat_id, text, reply_markup=reply_markup, parse_mode=None)
            last_menu_msgs[user_id] = new_msg.message_id
            return new_msg
        return None

async def delete_user_messages(chat_id: int, message_ids: list):
    """Safely deletes user messages."""
    for mid in message_ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=mid)
        except Exception:
            pass

@dp.message(Command("start"))
@dp.message(F.text == "Kembali")
async def cmd_start(message: Message, user_id: int = None):
    if user_id is None:
        user_id = message.from_user.id
        # Hanya hapus pesan jika ini adalah command asli dari user (bukan callback)
        if message.text and message.text != "Kembali":
            await delete_user_messages(message.chat.id, [message.message_id])
    
    settings = get_user_settings(user_id)
    text = (
        "🚀 **AI CONTENT CREATOR BOT PRO**\n\n"
        "Ubah video apapun menjadi konten viral secara otomatis dengan Gemini 1.5 Pro Modality.\n\n"
        "✨ **Status Fitur:**\n"
        "• 🧠 **AI Smart Mode**: " + ("✅ Aktif" if settings['mode'] == "ai" else "❌ Non-aktif") + "\n"
        "• 📱 **Auto Vertical**: " + ("✅ Aktif" if settings['vertical'] else "❌ Non-aktif") + "\n"
        "• 💬 **TikTok Subtitles**: " + ("✅ Aktif" if settings['subtitles'] else "❌ Non-aktif") + "\n\n"
        "⚙️ **Konfigurasi Saat Ini:**\n"
        f"• Target Durasi: `[ {settings['duration']} detik ]` 🔒\n"
        f"• Jumlah Klip: `[ {settings['max_clips']} klip ]` 🔒\n\n"
        "💡 **Cara Pakai:** Kirim video atau link ke sini. "
        "Kirim instruksi teks (misal: *'cari bagian lucu'*) sebelum mengirim video untuk panduan AI!"
    )
    
    kb = InlineKeyboardBuilder()
    kb.button(text="⏱ Durasi", callback_data="menu_dur")
    kb.button(text="📱 Vertical", callback_data="toggle_v")
    kb.button(text="💬 Subtitle", callback_data="toggle_s")
    kb.button(text="🎞 Jumlah", callback_data="menu_clips")
    kb.button(text="🤖 Mode AI", callback_data="toggle_ai")
    kb.button(text="🔄 Refresh", callback_data="menu_main")
    kb.adjust(2)
    
    await show_ui(message.chat.id, user_id, text, reply_markup=kb.as_markup())

@dp.callback_query(F.data == "menu_main")
async def callback_back(callback: CallbackQuery):
    await cmd_start(callback.message, user_id=callback.from_user.id)
    await callback.answer()

@dp.callback_query(F.data == "menu_dur")
async def callback_duration_menu(callback: CallbackQuery):
    kb = InlineKeyboardBuilder()
    for d in [15, 30, 60, 120, 180, 300, 600]:
        label = f"{d}s" if d < 60 else f"{d//60}m"
        kb.button(text=label, callback_data=f"set_dur_{d}")
    kb.button(text="⬅️ Kembali", callback_data="menu_main")
    kb.adjust(3)
    await show_ui(callback.message.chat.id, callback.from_user.id, "⏱ **Pilih Target Durasi Klip:**", reply_markup=kb.as_markup())
    await callback.answer()

@dp.callback_query(F.data.startswith("set_dur_"))
async def callback_set_duration(callback: CallbackQuery):
    user_id = callback.from_user.id
    dur = int(callback.data.split("_")[-1])
    settings = get_user_settings(user_id)
    settings['duration'] = dur
    await callback.answer(f"✅ Durasi diatur ke {dur}s (Saved)")
    await cmd_start(callback.message, user_id=user_id)

@dp.callback_query(F.data == "toggle_v")
async def callback_toggle_v(callback: CallbackQuery):
    user_id = callback.from_user.id
    settings = get_user_settings(user_id)
    settings['vertical'] = not settings['vertical']
    await callback.answer(f"📱 Vertical: {'ON' if settings['vertical'] else 'OFF'}")
    await cmd_start(callback.message, user_id=user_id)

@dp.callback_query(F.data == "toggle_s")
async def callback_toggle_s(callback: CallbackQuery):
    user_id = callback.from_user.id
    settings = get_user_settings(user_id)
    settings['subtitles'] = not settings['subtitles']
    await callback.answer(f"💬 Subtitle: {'ON' if settings['subtitles'] else 'OFF'}")
    await cmd_start(callback.message, user_id=user_id)

@dp.callback_query(F.data == "toggle_ai")
async def callback_toggle_ai(callback: CallbackQuery):
    user_id = callback.from_user.id
    settings = get_user_settings(user_id)
    settings['mode'] = "ai" if settings['mode'] != "ai" else "auto"
    await callback.answer(f"🤖 Mode: {settings['mode'].upper()}")
    await cmd_start(callback.message, user_id=user_id)

@dp.callback_query(F.data == "menu_clips")
async def callback_clips_menu(callback: CallbackQuery):
    kb = InlineKeyboardBuilder()
    for n in [1, 3, 5, 10, 15]:
        kb.button(text=f"{n} Klip", callback_data=f"set_n_{n}")
    kb.button(text="⬅️ Kembali", callback_data="menu_main")
    kb.adjust(3)
    await show_ui(callback.message.chat.id, callback.from_user.id, "🎞 **Pilih Jumlah Klip Maksimal:**", reply_markup=kb.as_markup())
    await callback.answer()

@dp.callback_query(F.data.startswith("set_n_"))
async def callback_set_clips(callback: CallbackQuery):
    user_id = callback.from_user.id
    n = int(callback.data.split("_")[-1])
    settings = get_user_settings(user_id)
    settings['max_clips'] = n
    await callback.answer(f"✅ Jumlah klip diatur ke {n} (Saved)")
    await cmd_start(callback.message, user_id=user_id)

@dp.message(F.video)
async def handle_video(message: Message):
    wait_msg = await message.answer("📥 Video diterima! Menambahkan ke antrean...")
    
    video_file = await bot.get_file(message.video.file_id)
    job_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(TEMP_DIR, f"{job_id}_telegram.mp4")
    
    await bot.download_file(video_file.file_path, file_path)
    
    settings = get_user_settings(message.from_user.id).copy()
    trigger_ids = [message.message_id]
    
    # Add custom instructions if any
    if message.from_user.id in user_prompts:
        prompt_data = user_prompts.pop(message.from_user.id)
        settings['custom_instructions'] = prompt_data['text']
        trigger_ids.append(prompt_data['msg_id'])
        
    # Add to queue
    await queue.put({
        'user_id': message.from_user.id,
        'chat_id': message.chat.id,
        'type': 'file',
        'data': file_path,
        'options': settings,
        'trigger_msg_ids': trigger_ids
    })
    
    mode_text = "Smart AI" if settings['mode'] == "ai" else "Visual Auto"
    await bot.edit_message_text(
        text=f"✅ Video masuk antrean! (Mode: `{mode_text}`)\n"
             f"Posisi: {queue.qsize()}\n\n"
             f"AI akan menganalisa visual & audio video Anda agar clipping lebih akurat. 🔍",
        chat_id=message.chat.id,
        message_id=wait_msg.message_id,
        parse_mode="Markdown"
    )
    
    # Update last_menu_msgs to this status message so worker can delete it
    last_menu_msgs[message.from_user.id] = wait_msg.message_id

# Session tracking for interactive selections
pending_sessions = {}

# Global store for custom prompts (temporary)
user_prompts = {}

@dp.message(F.text & ~F.text.regexp(r'^https?://') & ~F.text.regexp(r'^/') & ~F.text.in_({"⏱ Set Durasi", "📱 Toggle Vertical", "💬 Toggle Subtitle", "🎞 Jumlah Klip", "⚙️ Settings", "🤖 Toggle Mode AI", "Kembali", "15s", "30s", "60s", "90s", "2m", "3m", "5m", "10m", "1 Klip", "3 Klip", "5 Klip", "10 Klip", "15 Klip"}))
async def handle_custom_prompt(message: Message):
    user_id = message.from_user.id
    # Store message ID as well for deletion
    user_prompts[user_id] = {"text": message.text, "msg_id": message.message_id}
    
    text = f"🧠 **AI Instruction saved:**\n`{message.text}`\n\nKirim video/link sekarang untuk menerapkan instruksi ini."
    await show_ui(message.chat.id, user_id, text)

@dp.message(F.text.regexp(r'^https?://'))
async def handle_link(message: Message):
    url = message.text.strip()
    wait_msg = await message.answer("🔍 **Menganalisa video...** Mohon tunggu.", parse_mode="Markdown")
    
    info = await engine.get_video_info(url)
    if not info:
        return await wait_msg.edit_text("❌ Gagal mengambil info video. Pastikan link benar atau coba lagi nanti.")
    
    # Build Quality Keyboard
    kb = InlineKeyboardBuilder()
    for f in info['formats']:
        kb.button(text=f"{f['height']}p ({f['ext']})", callback_data=f"sel_q_{f['id']}")
    kb.button(text="Auto (Best)", callback_data="sel_q_best")
    kb.adjust(2)
    
    pending_sessions[message.from_user.id] = {
        'url': url,
        'info': info,
        'msg_ids': [wait_msg.message_id, message.message_id],
        'options': get_user_settings(message.from_user.id).copy()
    }
    
    # Store global prompt if any (experimental)
    if message.from_user.id in user_prompts:
        pending_sessions[message.from_user.id]['options']['custom_instructions'] = user_prompts.pop(message.from_user.id)
    
    duration_str = format_duration(info.get('duration', 0))
    await wait_msg.edit_text(
        f"🎬 **{info['title'][:50]}...**\n"
        f"⏱ Durasi Sumber: `{duration_str}`\n\n"
        f"Silakan pilih kualitas video:",
        reply_markup=kb.as_markup(),
        parse_mode="Markdown"
    )

@dp.callback_query(lambda c: c.data.startswith("sel_q_"))
async def process_quality_selection(callback: CallbackQuery):
    user_id = callback.from_user.id
    if user_id not in pending_sessions:
        return await callback.answer("Sesi kadaluarsa. Silakan kirim ulang link.", show_alert=True)
    
    quality_id = callback.data.replace("sel_q_", "")
    session = pending_sessions[user_id]
    session['options']['format_id'] = None if quality_id == "best" else quality_id
    
    # Check for subtitles
    info = session['info']
    if info.get('subtitles'):
        kb = InlineKeyboardBuilder()
        # Only show first 10 languages
        count = 0
        for code, name in info['subtitles'].items():
            kb.button(text=name[:15], callback_data=f"sel_s_{code}")
            count += 1
            if count >= 10: break
            
        kb.button(text="❌ Tanpa Subtitle", callback_data="sel_s_none")
        kb.adjust(2)
        
        await callback.message.edit_text(
            f"🌐 **Subtitles Terdeteksi!**\n\nPilih bahasa subtitle yang ingin digunakan (opsional):",
            reply_markup=kb.as_markup(),
            parse_mode="Markdown"
        )
    else:
        # No subtitles, jump to queue
        await finalize_link_selection(callback, "none")

@dp.callback_query(lambda c: c.data.startswith("sel_s_"))
async def process_subtitle_selection(callback: CallbackQuery):
    lang_code = callback.data.replace("sel_s_", "")
    await finalize_link_selection(callback, lang_code)

async def finalize_link_selection(callback: CallbackQuery, lang_code: str):
    user_id = callback.from_user.id
    if user_id not in pending_sessions:
        return await callback.answer("Sesi kadaluarsa.", show_alert=True)
        
    session = pending_sessions[user_id]
    if lang_code != "none":
        session['options']['subtitle_lang'] = lang_code
        session['options']['subtitles'] = True # Force burn subtitles if selected
    
    # Check if it was a full download request
    job_type = session['options'].get('type', 'url')
    
    # Add to queue
    await queue.put({
        'user_id': user_id,
        'chat_id': callback.message.chat.id,
        'type': job_type,
        'data': session['url'],
        'options': session['options'],
        'trigger_msg_ids': session['msg_ids']
    })
    
    # Success message (Editing the current selection message)
    await callback.message.edit_text(f"✅ Link masuk antrean! Posisi: {queue.qsize()}")
    last_menu_msgs[user_id] = callback.message.message_id

@dp.message(F.document)
async def handle_document(message: Message):
    if not is_admin(message.from_user.id):
        return await message.answer("🚫 Fitur ini hanya untuk admin.")
        
    doc = message.document
    file_name = doc.file_name.lower()
    valid_cookies = [
        "cookies.txt", "cookies_tiktok.txt", "cookies_youtube.txt",
        "cookies_facebook.txt", "cookies_instagram.txt", "cookies_twitter.txt"
    ]
    
    # Standardize filename to lowercase for internal use
    target_name = None
    for vc in valid_cookies:
        if file_name == vc:
            target_name = vc
            break
            
    if target_name:
        # Hapus file lama jika ada untuk memastikan refresh total
        old_size = 0
        if os.path.exists(target_name):
            old_size = os.path.getsize(target_name) / 1024
            os.remove(target_name)
            
        await bot.download_file((await bot.get_file(doc.file_id)).file_path, target_name)
        new_size = os.path.getsize(target_name) / 1024
        
        msg = (
            f"🔄 **Update Cookies Berhasil!**\n\n"
            f"📁 File: `{target_name}`\n"
            f"🗑️ Status: Cookie lama ({old_size:.1f} KB) telah dihapus.\n"
            f"✅ Status: Cookie baru ({new_size:.1f} KB) telah diterima dan diaktifkan."
        )
        await message.answer(msg, parse_mode="Markdown")
    else:
        await message.answer(f"📦 File `{doc.file_name}` diterima, tapi sistem hanya memproses file cookies: `cookies.txt`, `cookies_tiktok.txt`, or `cookies_youtube.txt`.", parse_mode="Markdown")

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    if not is_admin(message.from_user.id):
        return await message.answer("🚫 Maaf, Anda bukan admin.")
        
    # Check for cookie files
    cookie_status = []
    monitored_cookies = [
        "cookies.txt", "cookies_youtube.txt", "cookies_tiktok.txt",
        "cookies_facebook.txt", "cookies_instagram.txt", "cookies_twitter.txt"
    ]
    for cf in monitored_cookies:
        if os.path.exists(cf):
            size = os.path.getsize(cf) / 1024
            cookie_status.append(f"✅ `{cf}` ({size:.1f} KB)")
        else:
            cookie_status.append(f"❌ `{cf}` (Missing)")

    stats = (
        "📊 **Admin Statistics**\n\n"
        f"🔹 Antrean Aktif: `{queue.qsize()}`\n"
        f"🔹 Temp Folder: `{TEMP_DIR}`\n"
        f"🔹 Registered Admins: `{len(ADMIN_IDS)}`\n\n"
        "🍪 **Cookie Files Status:**\n" + "\n".join(cookie_status)
    )
    await message.answer(stats)

@dp.message(Command("update"))
async def cmd_update(message: Message):
    if not is_admin(message.from_user.id):
        return await message.answer("🚫 Akses ditolak.")
        
    status = await message.answer("🔄 **Mengecek pembaruan dari GitHub...**", parse_mode="Markdown")
    
    try:
        # Menjalankan git pull
        process = await asyncio.create_subprocess_shell(
            "git pull origin main",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        output = stdout.decode().strip() or stderr.decode().strip()
        
        update_log = (
            "✅ **Update Selesai!**\n\n"
            f"📝 **Log Output:**\n"
            f"```\n{output}\n```\n"
            "💡 *Catatan: Restart bot secara manual atau via PM2 jika ada perubahan pada library.*"
        )
        await status.edit_text(update_log, parse_mode="Markdown")
        
    except Exception as e:
        await status.edit_text(f"❌ **Update Gagal:**\n`{str(e)}`", parse_mode="Markdown")

@dp.message(Command("l"))
async def cmd_download_full(message: Message, command: CommandObject):
    if not command.args:
        return await message.answer("💡 **Cara penggunaan:** `/l [link]`\nContoh: `/l https://youtube.com/watch?v=...`", parse_mode="Markdown")
        
    url = command.args.strip()
    wait_msg = await message.answer("📥 **Mempersiapkan download video penuh...**", parse_mode="Markdown")
    
    info = await engine.get_video_info(url)
    if not info:
        return await wait_msg.edit_text("❌ Gagal mengambil info video. Pastikan link benar.")
        
    kb = InlineKeyboardBuilder()
    for f in info['formats']:
        kb.button(text=f"{f['height']}p ({f['ext']})", callback_data=f"sel_q_{f['id']}")
    kb.button(text="Auto (Best)", callback_data="sel_q_best")
    kb.adjust(2)
    
    pending_sessions[message.from_user.id] = {
        'url': url,
        'info': info,
        'msg_ids': [wait_msg.message_id, message.message_id],
        'options': {'type': 'full_download', 'vertical': False, 'subtitles': False}
    }
    
    await wait_msg.edit_text(
        f"📥 **Link Full Download Terdeteksi!**\n🎬 **{info['title'][:50]}...**\n\nSilakan pilih kualitas:",
        reply_markup=kb.as_markup(),
        parse_mode="Markdown"
    )

async def main():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    for _ in range(MAX_PARALLEL_JOBS):
        asyncio.create_task(worker())
    
    logger.info("Bot is starting...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")
