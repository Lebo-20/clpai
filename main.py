import os
import asyncio
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder

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
            "mode": "auto",
            "subtitles": False,
            "vertical": True,
            "max_clips": 3
        }
    return user_settings[user_id]

def is_admin(user_id: int) -> bool:
    return not ADMIN_IDS or user_id in ADMIN_IDS

def get_progress_bar(percentage):
    filled = int(percentage / 10)
    bar = "██" * filled + "░░" * (10 - filled)
    return f"{bar} {percentage}%"

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
        
        # Initial status message
        status_msg = await bot.send_message(chat_id, f"📡 **Job `{job_id}`**: Menghubungkan ke server...")

        async def update_progress(percentage, stage_msg):
            progress_text = (
                f"🚀 **Processing Job**: `{job_id}`\n"
                f"📊 Stage: `{stage_msg}`\n\n"
                f"{get_progress_bar(percentage)}"
            )
            try:
                # Update but not too fast to avoid 429
                await bot.edit_message_text(text=progress_text, chat_id=chat_id, message_id=status_msg.message_id, parse_mode="Markdown")
            except Exception:
                pass

        input_path = None
        try:
            logger.info(f"Processing job {job_id} for user {user_id}")
            
            # STAGE 1: Download (0-30%)
            await update_progress(5, "Downloading video...")
            if input_type == "url":
                input_path = await engine.download_video(input_data, job_id)
            else:
                input_path = input_data
            await update_progress(30, "Download complete.")

            # STAGE 2-5: Processing (30-95%)
            clips = await engine.create_clips(input_path, job_id, options, progress_callback=update_progress)

            # STAGE 6: Upload (95-100%)
            if clips:
                await update_progress(96, "Uploading results...")
                # Filename final_highlight.mp4 is always at the end
                for i, clip_path in enumerate(clips):
                    is_merged = os.path.basename(clip_path) == "final_highlight.mp4"
                    video = FSInputFile(clip_path)
                    
                    if is_merged:
                        caption = f"🎬 **FINAL HIGHLIGHT** (ID `{job_id}`)\n\nTop 15 scene terbaik pilihan AI."
                    else:
                        caption = f"🎬 **Clip {i+1}** (ID `{job_id}`)"
                    
                    await bot.send_video(chat_id, video, caption=caption, parse_mode="Markdown")
                    await asyncio.sleep(1)
                
                await update_progress(100, "Done! Semua klip telah dikirim.")
            else:
                await update_progress(100, "Gagal: Tidak ada klip ditemukan.")

        except Exception as e:
            logger.error(f"Error in worker for job {job_id}: {e}")
            await bot.edit_message_text(text=f"❌ Job `{job_id}` gagal: {str(e)[:100]}", chat_id=chat_id, message_id=status_msg.message_id)
        finally:
            if input_path:
                await engine.cleanup(input_path)
            queue.task_done()
            logger.info(f"Finished job {job_id}")

# Store last menu message ID to clean up
last_menu_msgs = {}

async def clean_send(message: Message, text: str, reply_markup=None, parse_mode="Markdown"):
    """Deletes previous menu and user trigger to keep chat clean."""
    user_id = message.from_user.id
    
    # 1. Delete user's command
    try:
        await message.delete()
    except Exception:
        pass
        
    # 2. Delete last bot menu if exists
    if user_id in last_menu_msgs:
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=last_menu_msgs[user_id])
        except Exception:
            pass
            
    # 3. Send new message and store ID
    new_msg = await bot.send_message(
        chat_id=message.chat.id, 
        text=text, 
        reply_markup=reply_markup, 
        parse_mode=parse_mode
    )
    last_menu_msgs[user_id] = new_msg.message_id
    return new_msg

@dp.message(Command("start"))
async def cmd_start(message: Message):
    settings = get_user_settings(message.from_user.id)
    text = (
        "👋 **Selamat datang di AI Video Clipper Pro!**\n\n"
        "Ubah video panjang menjadi konten pendek viral secara otomatis.\n\n"
        "⚙️ **Konfigurasi Aktif:**\n"
        f"• Mode: `{settings['mode']}`\n"
        f"• Durasi: `{settings['duration']}s`\n"
        f"• Subtitle: `{'On' if settings['subtitles'] else 'Off'}`\n"
        f"• Vertical: `{'Yes' if settings['vertical'] else 'No'}`\n\n"
        "🎥 **Kirim Link atau File Video untuk mulai.**"
    )
    kb = [
        [KeyboardButton(text="⏱ Set Durasi"), KeyboardButton(text="📱 Toggle Vertical")],
        [KeyboardButton(text="💬 Toggle Subtitle"), KeyboardButton(text="⚙️ Settings")]
    ]
    await clean_send(message, text, reply_markup=ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True))

@dp.message(F.text == "⏱ Set Durasi")
async def set_duration_menu(message: Message):
    kb = [
        [KeyboardButton(text="15s"), KeyboardButton(text="30s"), KeyboardButton(text="60s"), KeyboardButton(text="90s")],
        [KeyboardButton(text="2m"), KeyboardButton(text="3m"), KeyboardButton(text="5m")],
        [KeyboardButton(text="10m"), KeyboardButton(text="Kembali")]
    ]
    await clean_send(message, "Pilih durasi klip target (s=detik, m=menit):", reply_markup=ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True))

@dp.message(F.text.regexp(r'^(\d+)(s|m)$'))
async def process_duration_set(message: Message):
    text = message.text
    dur = int(text.replace("s", "")) if text.endswith("s") else int(text.replace("m", "")) * 60
    settings = get_user_settings(message.from_user.id)
    settings['duration'] = dur
    # Return to start menu after setting
    await cmd_start(message)

@dp.message(F.text == "📱 Toggle Vertical")
async def toggle_vertical(message: Message):
    settings = get_user_settings(message.from_user.id)
    settings['vertical'] = not settings['vertical']
    await cmd_start(message)

@dp.message(F.text == "💬 Toggle Subtitle")
async def toggle_subs(message: Message):
    settings = get_user_settings(message.from_user.id)
    settings['subtitles'] = not settings['subtitles']
    await cmd_start(message)

@dp.message(F.text == "⚙️ Settings")
async def cmd_settings(message: Message):
    settings = get_user_settings(message.from_user.id)
    kb = [
        [KeyboardButton(text="🤖 Toggle Mode AI")],
        [KeyboardButton(text="Kembali")]
    ]
    status = "Smart AI" if settings['mode'] == "ai" else "Visual Auto"
    await clean_send(message, f"⚙️ **Pengaturan Mode Clipping**\n\nMode Saat Ini: `{status}`", reply_markup=ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True))

@dp.message(F.text == "🤖 Toggle Mode AI")
async def toggle_ai_mode(message: Message):
    settings = get_user_settings(message.from_user.id)
    settings['mode'] = "ai" if settings['mode'] != "ai" else "auto"
    await cmd_start(message)

@dp.message(F.text == "Kembali")
async def cmd_back(message: Message):
    await cmd_start(message)

@dp.message(F.video)
async def handle_video(message: Message):
    wait_msg = await message.answer("📥 Video diterima! Menambahkan ke antrean...")
    
    video_file = await bot.get_file(message.video.file_id)
    job_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(TEMP_DIR, f"{job_id}_telegram.mp4")
    
    await bot.download_file(video_file.file_path, file_path)
    
    settings = get_user_settings(message.from_user.id).copy()
    await queue.put({
        'user_id': message.from_user.id,
        'chat_id': message.chat.id,
        'type': 'file',
        'data': file_path,
        'options': settings
    })
    
    await bot.edit_message_text(text=f"✅ Video masuk antrean. Posisi: {queue.qsize()}", chat_id=message.chat.id, message_id=wait_msg.message_id)

@dp.message(F.text.regexp(r'^https?://'))
async def handle_link(message: Message):
    wait_msg = await message.answer("🔗 Link terdeteksi! Menambahkan ke antrean...")
    
    settings = get_user_settings(message.from_user.id).copy()
    await queue.put({
        'user_id': message.from_user.id,
        'chat_id': message.chat.id,
        'type': 'url',
        'data': message.text,
        'options': settings
    })
    
    await bot.edit_message_text(text=f"✅ Link masuk antrean. Posisi: {queue.qsize()}", chat_id=message.chat.id, message_id=wait_msg.message_id)

@dp.message(F.document)
async def handle_document(message: Message):
    if not is_admin(message.from_user.id):
        return await message.answer("🚫 Fitur ini hanya untuk admin.")
        
    doc = message.document
    valid_cookies = ["cookies.txt", "cookies_tiktok.txt", "cookies_youtube.txt"]
    
    if doc.file_name in valid_cookies:
        await bot.download_file((await bot.get_file(doc.file_id)).file_path, doc.file_name)
        await message.answer(f"✅ **{doc.file_name}** berhasil diperbarui dan dipisahkan sesuai platform!", parse_mode="Markdown")
    else:
        await message.answer(f"📦 File `{doc.file_name}` diterima, tapi sistem hanya memproses file cookies (`cookies.txt`, `cookies_tiktok.txt`, `cookies_youtube.txt`).", parse_mode="Markdown")

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    if not is_admin(message.from_user.id):
        return await message.answer("🚫 Maaf, Anda bukan admin.")
        
    stats = (
        "📊 **Admin Statistics**\n\n"
        f"🔹 Antrean Aktif: `{queue.qsize()}`\n"
        f"🔹 Temp Folder: `{TEMP_DIR}`\n"
        f"🔹 Registered Admins: `{len(ADMIN_IDS)}`"
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

async def main():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
        
    for _ in range(MAX_PARALLEL_JOBS):
        asyncio.create_task(worker())
    
    logger.info("Bot is starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")
