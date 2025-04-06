import logging
import os
import asyncio
import re
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv
from twitchio.ext import commands
from googletrans import Translator, LANGUAGES
import langdetect
import spacy
import time

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 環境変数の読み込み ---
load_dotenv()

# --- 環境変数と直接設定値 ---
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN")
CHANNEL = os.getenv("TWITCH_CHANNEL")
TRANSLATION_CACHE_SIZE = int(os.getenv("TRANSLATION_CACHE_SIZE", 100))
LANGUAGE_DETECTION_CACHE_SIZE = int(os.getenv("LANGUAGE_DETECTION_CACHE_SIZE", 200))

# --- 直接設定値 ---
IGNORE_USERS = ["nightbot", "streamelements"]  # 無視するユーザー名 (小文字)
IGNORE_LINES = ["!command", "http://"]  # この文字列を含む行は無視
IGNORE_WORDS = ["spam_word", "bad_word"]  # この単語を含むメッセージは無視 (小文字)
MAX_TRANSLATION_LENGTH = 500  # 翻訳するメッセージの最大長
MIN_TRANSLATION_LENGTH = 5  # これより短いメッセージは翻訳しない
LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.70  # 言語検出の最低信頼度
JAPANESE_CHARACTER_THRESHOLD = 0.3  # 日本語と判定されても、日本語文字の割合がこれ未満なら翻訳試行 (0.0-1.0)
COMMON_SLANGS = ["gg", "lol", "pog", "kappa", "lmao", "omegalul", "kekw"]  # 翻訳しない定型句 (小文字)
IGNORE_URLS = True  # メッセージ中のURLを翻訳対象から除くか
IGNORE_MENTIONS = True  # メッセージ中の @メンション を翻訳対象から除くか
TRANSLATION_RETRY_COUNT = 3
TRANSLATION_RETRY_DELAY = 1
MAX_WORKERS = 3  # 非同期翻訳の最大同時実行数
GOOGLETRANS_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

#--- 必須の環境変数が設定されているか確認 ---
#if not all([CLIENT_ID, ACCESS_TOKEN, CHANNEL]):
#    raise ValueError("Required environment variables are missing: TWITCH_CLIENT_ID, TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL")
#--- 必須の環境変数が設定されているか確認 ---
#--- Google Translate の初期化 ---
translator_pool = []
for _ in range(MAX_WORKERS):
    try:
        # ユーザーエージェントを設定して異なるセッションを作成
        translator = Translator(user_agent=GOOGLETRANS_USER_AGENT)
        translator_pool.append(translator)
    except Exception as e:
        logger.error(f"Failed to initialize a translator: {e}")

logger.info(f"Initialized {len(translator_pool)} translators in pool")

# --- spaCyの日本語モデルをロード ---
try:
    nlp = spacy.load("ja_core_news_sm")
    logger.info("Loaded spaCy Japanese model.")
except OSError:
    logger.warning("spaCy Japanese model 'ja_core_news_sm' not found. spaCy language detection will be disabled.")
    nlp = None

# --- Google Translateがサポートする言語コードのリスト ---
SUPPORTED_LANGUAGES = LANGUAGES
logger.info(f"Supported languages: {len(SUPPORTED_LANGUAGES)}")

# --- LRUキャッシュ ---
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        logger.info(f"LRUCache initialized (capacity: {capacity}).")

    def get(self, key: Any) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        if self.capacity == 0:
            return
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self) -> None:
        self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)

translation_cache = LRUCache(TRANSLATION_CACHE_SIZE)
language_detection_cache = LRUCache(LANGUAGE_DETECTION_CACHE_SIZE)

# --- セマフォによる同時実行数制限 ---
translation_semaphore = asyncio.Semaphore(MAX_WORKERS)
translator_lock = asyncio.Lock()
translator_index = 0

# --- 翻訳関数 ---
async def get_translator():
    global translator_index
    async with translator_lock:
        if not translator_pool:
            # 緊急対応: プールが空なら新しいトランスレータを作成
            logger.warning("Translator pool is empty, creating a new translator.")
            translator = Translator(user_agent=GOOGLETRANS_USER_AGENT)
            translator_index = 0
            return translator
            
        translator = translator_pool[translator_index]
        translator_index = (translator_index + 1) % len(translator_pool)
        return translator

async def translate_with_retry(text: str, target_language: str = 'ja') -> Optional[Tuple[str, str]]:
    if not text:
        return None
    
    for attempt in range(TRANSLATION_RETRY_COUNT + 1):
        try:
            translator = await get_translator()
            loop = asyncio.get_event_loop()
            translation = await loop.run_in_executor(
                None,
                lambda: translator.translate(text, dest=target_language)
            )
            
            if translation and translation.text:
                result = (translation.text, translation.src)
                return result
            else:
                logger.warning(f"Invalid translation result: {translation}")
                if attempt < TRANSLATION_RETRY_COUNT:
                    delay = TRANSLATION_RETRY_DELAY * (2 ** attempt)  # 指数バックオフ
                    logger.info(f"Retrying translation in {delay}s (attempt {attempt + 1}/{TRANSLATION_RETRY_COUNT})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Translation failed after {TRANSLATION_RETRY_COUNT} attempts: '{text}'")
                    return None
        except Exception as e:
            logger.error(f"Translation error for '{text}': {e}")
            if attempt < TRANSLATION_RETRY_COUNT:
                delay = TRANSLATION_RETRY_DELAY * (2 ** attempt)  # 指数バックオフ
                logger.info(f"Retrying translation in {delay}s (attempt {attempt + 1}/{TRANSLATION_RETRY_COUNT})")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Translation failed after {TRANSLATION_RETRY_COUNT} attempts: '{text}'")
                return None
    
    return None

async def translate(text: str, target_language: str = 'ja') -> Optional[Tuple[str, str]]:
    if not text:
        return None
    
    # キャッシュをチェック
    cache_key = (text, target_language)
    cached_result = translation_cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # セマフォを使用して同時実行数を制限
    async with translation_semaphore:
        try:
            result = await translate_with_retry(text, target_language)
            if result:
                translation_cache.put(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Translation failed for '{text}': {e}", exc_info=True)
            return None

# --- 言語検出関数 ---
def detect_language_spacy(text: str) -> Optional[str]:
    if not nlp:
        return None
    try:
        doc = nlp(text)
        if doc.lang_:
            return doc.lang_
        return None
    except Exception as e:
        logger.exception(f"spaCy language detection error: {e}")
        return None

def detect_language_with_confidence(text: str) -> Optional[Tuple[str, float]]:
    try:
        langs = langdetect.detect_langs(text)
        if langs:
            best_lang = langs[0]
            return best_lang.lang, best_lang.prob
        return None, 0.0
    except langdetect.LangDetectException:
        return None, 0.0
    except Exception as e:
        logger.exception(f"Unexpected error in langdetect: {e}")
        return None, 0.0

async def detect_language(text: str) -> Tuple[Optional[str], float]:
    """複数の方法を使用して言語を検出し、最も信頼性の高い結果を返す"""
    if not text:
        return None, 0.0
    
    # キャッシュをチェック
    cached_result = language_detection_cache.get(text)
    if cached_result:
        return cached_result
    
    lang, confidence = None, 0.0
    
    # まずlangdetectを試す
    try:
        lang_result, conf = detect_language_with_confidence(text)
        if lang_result:
            lang = lang_result
            confidence = conf
    except Exception as e:
        logger.debug(f"langdetect failed: {e}")
    
    # 信頼度が低い場合はspaCyも試す
    if (not lang or confidence < 0.5) and nlp:
        try:
            spacy_lang = detect_language_spacy(text)
            if spacy_lang:
                lang = spacy_lang
                confidence = 0.8  # spaCyには明示的な信頼度がないため仮定値
        except Exception as e:
            logger.debug(f"spaCy detection failed: {e}")
    
    result = (lang, confidence)
    language_detection_cache.put(text, result)
    return result

# --- ヘルパー関数 ---
def contains_significant_japanese(text: str, threshold: float) -> bool:
    if not text:
        return False
    japanese_chars = 0
    total_chars = 0
    for char in text:
        if char.strip():
            total_chars += 1
            if ('\u3040' <= char <= '\u309F') or \
               ('\u30A0' <= char <= '\u30FF') or \
               ('\u4E00' <= char <= '\u9FFF') or \
               ('\uF900' <= char <= '\uFAFF') or \
               ('\u3400' <= char <= '\u4DBF'):
                japanese_chars += 1
    if total_chars == 0:
        return False
    ratio = japanese_chars / total_chars
    return ratio >= threshold

def remove_urls_mentions(text: str) -> Tuple[str, List[str], List[str]]:
    urls = []
    mentions = []
    processed_text = text

    if IGNORE_URLS:
        # より堅牢なURL検出パターン
        url_pattern = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))'
        urls = re.findall(url_pattern, processed_text)
        processed_text = re.sub(url_pattern, '[URL]', processed_text)

    if IGNORE_MENTIONS:
        mention_pattern = r'(@[a-zA-Z0-9_]+)'
        mentions = re.findall(mention_pattern, processed_text)
        processed_text = re.sub(mention_pattern, '[MENTION]', processed_text)

    return processed_text.strip(), urls, mentions

def should_translate_message(user: str, content: str) -> bool:
    """メッセージを翻訳すべきかどうかを判断する"""
    if user in IGNORE_USERS:
        return False
    if any(line in content for line in IGNORE_LINES):
        return False
    if len(content) < MIN_TRANSLATION_LENGTH:
        return False
    if len(content) > MAX_TRANSLATION_LENGTH:
        return False
    if any(word in content.lower() for word in IGNORE_WORDS):
        return False
    if content.lower() in COMMON_SLANGS:
        return False
    return True

# --- Botクラス ---
class Bot(commands.Bot):
    def __init__(self):
        super().__init__(token=ACCESS_TOKEN, client_id=CLIENT_ID, prefix='!', initial_channels=[CHANNEL])
        self.translation_enabled = True
        self.stats = {
            "messages_processed": 0,
            "messages_translated": 0,
            "translation_errors": 0,
            "start_time": time.time()
        }

    async def event_ready(self):
        logger.info(f'Logged in | {self.nick}')
        await self.wait_for_ready()
        try:
            connected_channels = self.connected_channels
            if connected_channels:
                logger.info(f"Connected channels: {[ch.name for ch in connected_channels]}")
            else:
                logger.warning("Not connected to any channels.")
        except AttributeError as e:
            logger.error(f"Error getting startup information: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during startup: {e}", exc_info=True)
        logger.info("Bot is ready!")

    async def event_message(self, message):
        if message.echo:
            return
        self.stats["messages_processed"] += 1
        if message.content.startswith('!'):
            await self.handle_commands(message)
            return
        await self.process_message(message)

    async def process_message(self, message):
        user = message.author.name.lower()
        original_content = message.content.strip()
        channel_name = message.channel.name

        if not self.translation_enabled:
            return
        
        if not should_translate_message(user, original_content):
            return

        content_to_process, urls, mentions = remove_urls_mentions(original_content)
        if not content_to_process and (urls or mentions):
            return

        detected_lang, confidence = await detect_language(content_to_process)
        
        should_translate = False
        if detected_lang is None:
            should_translate = True
        elif detected_lang == 'ja' and not contains_significant_japanese(content_to_process, JAPANESE_CHARACTER_THRESHOLD):
            should_translate = True
        elif confidence < LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
            should_translate = True
        elif detected_lang != 'ja':  # 日本語以外は翻訳
            should_translate = True
        else:
            return

        if should_translate:
            try:
                translation_result = await translate(content_to_process, 'ja')
                if translation_result:
                    translated_text, source_lang = translation_result
                    if translated_text.strip() != content_to_process.strip():
                        display_lang = source_lang if source_lang != "unknown" else detected_lang if detected_lang else "?"
                        await message.channel.send(f"{user} ({display_lang.upper()}): {translated_text}")
                        self.stats["messages_translated"] += 1
            except Exception as e:
                logger.error(f"Translation error: {e}", exc_info=True)
                self.stats["translation_errors"] += 1

    @commands.command(name='ping')
    async def ping(self, ctx: commands.Context):
        await ctx.send(f'Pong! ({round(self.latency * 1000)}ms)')

    @commands.command(name='togglet')
    async def toggle_translation(self, ctx: commands.Context):
        self.translation_enabled = not self.translation_enabled
        status = "enabled" if self.translation_enabled else "disabled"
        await ctx.send(f"Automatic translation is now {status}.")
    
    @commands.command(name='jatoen')
    async def translate_ja_to_en(self, ctx: commands.Context, *, text: str):
        """Translate Japanese text to English."""
        try:
            cleaned_text, _, _ = remove_urls_mentions(text)
            if not cleaned_text:
                await ctx.send("No text to translate.")
                return
            translation_result = await translate(cleaned_text, 'en')
            if translation_result:
                translated_text, source_lang = translation_result
                await ctx.send(f"{ctx.author.name} (JA->EN): {translated_text}")
            else:
                await ctx.send("Translation failed.")
        except Exception as e:
            logger.error(f"jatoen command error: {e}", exc_info=True)
            await ctx.send(f"Translation error: {e}")

    @commands.command(name='clear_cache')
    async def clear_cache(self, ctx: commands.Context):
        """Clear translation cache."""
        if not (ctx.author.is_mod or ctx.author.name.lower() == CHANNEL.lower()):
            await ctx.send("This command is only available to the streamer or moderators.")
            return
        
        translation_cache_size = len(translation_cache)
        language_cache_size = len(language_detection_cache)
        
        translation_cache.clear()
        language_detection_cache.clear()
        
        await ctx.send(f'{ctx.author.name}, caches cleared. (Translation: {translation_cache_size}, Language detection: {language_cache_size} entries)')
        logger.info(f"Caches cleared by {ctx.author.name}")

    @commands.command(name='stats')
    async def show_stats(self, ctx: commands.Context):
        """Show bot statistics."""
        uptime = time.time() - self.stats["start_time"]
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        uptime_str = f"{hours}h {minutes}m {seconds}s"
        messages_per_minute = self.stats["messages_processed"] / (uptime / 60) if uptime > 0 else 0
        
        stats_msg = (
            f"Bot Statistics:\n"
            f"- Uptime: {uptime_str}\n"
            f"- Messages processed: {self.stats['messages_processed']}\n"
            f"- Messages translated: {self.stats['messages_translated']}\n"
            f"- Translation errors: {self.stats['translation_errors']}\n"
            f"- Messages per minute: {messages_per_minute:.1f}\n"
            f"- Translation cache size: {len(translation_cache)}/{TRANSLATION_CACHE_SIZE}\n"
            f"- Language detection cache size: {len(language_detection_cache)}/{LANGUAGE_DETECTION_CACHE_SIZE}"
        )
        
        await ctx.send(stats_msg)
        
    @commands.command(name='supported')
    async def show_supported_languages(self, ctx: commands.Context):
        """Show supported languages."""
        if not SUPPORTED_LANGUAGES:
            await ctx.send("Failed to retrieve supported languages.")
            return
            
        # 一部の主要言語のみ表示
        major_languages = {
            'en': 'English', 
            'ja': 'Japanese', 
            'zh-cn': 'Chinese (Simplified)',
            'ko': 'Korean',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ru': 'Russian'
        }
        
        langs = [f"{code}: {name}" for code, name in major_languages.items() if code in SUPPORTED_LANGUAGES]
        
        await ctx.send(f"Major supported languages: {', '.join(langs)} (and {len(SUPPORTED_LANGUAGES) - len(langs)} more)")
        
    @commands.command(name='restart_translators')
    async def restart_translators(self, ctx: commands.Context):
        """Restart translator pool (admin only)."""
        if not (ctx.author.is_mod or ctx.author.name.lower() == CHANNEL.lower()):
            await ctx.send("This command is only available to the streamer or moderators.")
            return
            
        global translator_pool
        global translator_index
        
        # トランスレータープールを再起動
        async with translator_lock:
            translator_pool = []
            translator_index = 0
            for _ in range(MAX_WORKERS):
                try:
                    translator = Translator(user_agent=GOOGLETRANS_USER_AGENT)
                    translator_pool.append(translator)
                except Exception as e:
                    logger.error(f"Failed to initialize a translator: {e}")
                    
        await ctx.send(f"Translator pool restarted. New size: {len(translator_pool)}/{MAX_WORKERS}")
        logger.info(f"Translator pool restarted by {ctx.author.name}")

if __name__ == "__main__":
    if not translator_pool:
        logger.warning("No translators initialized in pool. Translation may not work.")
        
    if nlp is None:
        logger.warning("spaCy Japanese model not loaded. spaCy language detection will not be used.")

    bot = Bot()
    bot.run()
