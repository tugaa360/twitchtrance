import logging
import os
import asyncio
import re
from collections import OrderedDict
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from twitchio.ext import commands
import googletrans  # googletrans モジュール全体をインポート
from googletrans import Translator, __version__, LANGUAGES
import langdetect
import config  # config.py を読み込む
import spacy
import time

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 環境変数の読み込み ---
load_dotenv()

# --- 環境変数から設定を取得 ---
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN")
CHANNEL = os.getenv("TWITCH_CHANNEL")
TRANSLATION_CACHE_SIZE = int(os.getenv("TRANSLATION_CACHE_SIZE", 100))

# --- config.py から設定を読み込み (存在しない場合のデフォルト値も設定) ---
IGNORE_USERS = [user.lower() for user in getattr(config, 'IGNORE_USERS', [])]
IGNORE_LINES = getattr(config, 'IGNORE_LINES', [])
IGNORE_WORDS = [word.lower() for word in getattr(config, 'IGNORE_WORDS', [])]
MAX_TRANSLATION_LENGTH = getattr(config, 'MAX_TRANSLATION_LENGTH', 500)
MIN_TRANSLATION_LENGTH = getattr(config, 'MIN_TRANSLATION_LENGTH', 5)
LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = getattr(config, 'LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD', 0.80)
JAPANESE_CHARACTER_THRESHOLD = getattr(config, 'JAPANESE_CHARACTER_THRESHOLD', 0.3)
COMMON_SLANGS = [slang.lower() for slang in getattr(config, 'COMMON_SLANGS', ["gg", "lol", "pog", "kappa", "lmao", "omegalul", "kekw"])]
IGNORE_URLS = getattr(config, 'IGNORE_URLS', True)
IGNORE_MENTIONS = getattr(config, 'IGNORE_MENTIONS', True)
TRANSLATION_RETRY_COUNT = getattr(config, 'TRANSLATION_RETRY_COUNT', 3)
TRANSLATION_RETRY_DELAY = getattr(config, 'TRANSLATION_RETRY_DELAY', 1)

# --- 必須の環境変数が設定されているか確認 ---
if not all([CLIENT_ID, ACCESS_TOKEN, CHANNEL]):
    raise ValueError("必要な環境変数が不足しています: TWITCH_CLIENT_ID, TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL")

# --- Google Translate の初期化 ---
translator = Translator()
logger.info(f"googletrans のバージョン: {__version__}")

# --- spaCyの日本語モデルをロード ---
try:
    nlp = spacy.load("ja_core_news_sm")
except OSError:
    logger.warning("spaCyの日本語モデル 'ja_core_news_sm' が見つかりません。spaCyによる言語検出は無効になります。")
    nlp = None

# --- Google Translateがサポートする言語コードのリスト ---
SUPPORTED_LANGUAGES = LANGUAGES

# --- LRUキャッシュ ---
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        logger.info(f"LRUCache を初期化しました (capacity: {capacity})。")

    def get(self, key):
        if key in self.cache:
            logger.debug(f"キャッシュヒット: {key}")
            self.cache.move_to_end(key)
            return self.cache[key]
        logger.debug(f"キャッシュミス: {key}")
        return None

    def put(self, key, value):
        if self.capacity == 0: return # キャッシュが無効なら何もしない
        logger.debug(f"キャッシュに保存: {key}")
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            removed_key, _ = self.cache.popitem(last=False)
            logger.debug(f"キャッシュから削除 (LRU): {removed_key}")
        self.cache[key] = value

translation_cache = LRUCache(TRANSLATION_CACHE_SIZE)

# --- 翻訳関数 ---
async def translate(text: str, target_language: str = 'ja') -> Optional[Tuple[str, str]]:
    """テキストを翻訳し、翻訳結果とソース言語を返す。"""
    if not text:
        return None
    cache_key = (text, target_language)
    cached_result = translation_cache.get(cache_key)
    if cached_result:
        return cached_result

    for attempt in range(TRANSLATION_RETRY_COUNT + 1):
        try:
            logger.debug(f"翻訳API呼び出し試行 {attempt + 1}/{TRANSLATION_RETRY_COUNT + 1}: '{text}' -> {target_language}")
            translation = translator.translate(text, dest=target_language)

            if translation and translation.text:
                logger.debug(f"翻訳成功: '{text}' ({translation.src}) -> '{translation.text}' ({target_language})")
                result = (translation.text, translation.src)
                translation_cache.put(cache_key, result)
                return result
            else:
                logger.warning(f"翻訳結果が無効です: {translation}")
                if attempt < TRANSLATION_RETRY_COUNT:
                    await asyncio.sleep(TRANSLATION_RETRY_DELAY)
                else:
                    logger.error(f"翻訳結果が無効なため、最終的に失敗とします: '{text}'")
                    return None

        except googletrans.exceptions.TooManyRequests as e:  # 修正: googletrans.exceptions.TooManyRequests
            logger.warning(f"翻訳APIレート制限 (試行 {attempt + 1}/{TRANSLATION_RETRY_COUNT + 1}): {e}")
            if attempt < TRANSLATION_RETRY_COUNT:
                await asyncio.sleep(TRANSLATION_RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"レート制限により翻訳失敗: '{text}'")
                return None
        except googletrans.exceptions.NotTranslated as e:  # 修正: googletrans.exceptions.NotTranslated
            logger.warning(f"翻訳APIが空応答 (試行 {attempt + 1}/{TRANSLATION_RETRY_COUNT + 1}): '{text}'. リトライします。")
            if attempt < TRANSLATION_RETRY_COUNT:
                await asyncio.sleep(TRANSLATION_RETRY_DELAY)
            else:
                logger.error(f"翻訳APIが空応答を繰り返し失敗: '{text}'")
                return None
        except Exception as e:
            logger.error(f"予期せぬ翻訳エラーが発生しました (試行 {attempt + 1}/{TRANSLATION_RETRY_COUNT + 1}): '{text}': {e}", exc_info=True)
            return None
    return None

# --- 言語検出関数 ---
def detect_language_spacy(text: str) -> Optional[str]:
    """spaCyを使用して言語を検出する"""
    if not nlp:
        return None
    try:
        doc = nlp(text)
        if doc.lang_:
            logger.debug(f"spaCy detected: {doc.lang_}")
            return doc.lang_
        return None
    except Exception as e:
        logger.exception(f"spaCy言語検出でエラー：{e}")
        return None

def detect_language_with_confidence(text: str) -> Optional[Tuple[str, float]]:
    """langdetectを使用して言語とその信頼度を検出する"""
    try:
        langs = langdetect.detect_langs(text)
        if langs:
            best_lang = langs[0]
            logger.debug(f"langdetect detected: {best_lang.lang} (prob: {best_lang.prob:.4f})")
            return best_lang.lang, best_lang.prob
        return None, 0.0
    except langdetect.LangDetectException:
        logger.debug(f"langdetect cannot detect language: '{text}' (may be too short or no specific language features)")
        return None, 0.0
    except Exception as e:
        logger.exception(f"langdetectで予期せぬエラー：{e}")
        return None, 0.0

# --- ヘルパー関数 ---
def contains_significant_japanese(text: str, threshold: float) -> bool:
    """メッセージ中の日本語文字(ひらがな、カタカナ、漢字)の割合が閾値以上か判定"""
    if not text: return False
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

    if total_chars == 0: return False
    ratio = japanese_chars / total_chars
    logger.debug(f"日本語文字の割合: {ratio:.2f} (閾値: {threshold})")
    return ratio >= threshold

def remove_urls_mentions(text: str) -> Tuple[str, List[str], List[str]]:
    """テキストからURLとメンションを削除し、元の要素をリストで返す"""
    urls = []
    mentions = []
    processed_text = text

    if IGNORE_URLS:
        url_pattern = r'(https?://[^\s]+)'
        urls = re.findall(url_pattern, processed_text)
        processed_text = re.sub(url_pattern, '[URL]', processed_text)
        logger.debug(f"削除されたURL: {urls}")

    if IGNORE_MENTIONS:
        mention_pattern = r'(@[a-zA-Z0-9_]+)'
        mentions = re.findall(mention_pattern, processed_text)
        processed_text = re.sub(mention_pattern, '[MENTION]', processed_text)
        logger.debug(f"削除されたメンション: {mentions}")

    return processed_text.strip(), urls, mentions

# --- Botクラス ---
class Bot(commands.Bot):
    def __init__(self):
        super().__init__(token=ACCESS_TOKEN, client_id=CLIENT_ID, prefix='!', initial_channels=[CHANNEL])
        self.translation_enabled = True
        logger.info("Bot を初期化しました。")
        logger.info(f"無視するユーザー: {IGNORE_USERS}")
        logger.info(f"無視する行パターン: {IGNORE_LINES}")
        logger.info(f"無視する単語: {IGNORE_WORDS}")
        logger.info(f"翻訳しない定型句: {COMMON_SLANGS}")
        logger.info(f"最小翻訳長: {MIN_TRANSLATION_LENGTH}")
        logger.info(f"最大翻訳長: {MAX_TRANSLATION_LENGTH}")
        logger.info(f"言語検出信頼度閾値: {LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD}")
        logger.info(f"日本語文字割合閾値: {JAPANESE_CHARACTER_THRESHOLD}")
        logger.info(f"URLを無視: {IGNORE_URLS}")
        logger.info(f"メンションを無視: {IGNORE_MENTIONS}")

    async def event_ready(self):
        logger.info(f'ログインしました | {self.nick}')
        await self.wait_for_ready()
        try:
            connected_channels = self.connected_channels
            if connected_channels:
                logger.info(f"接続中のチャンネル: {[ch.name for ch in connected_channels]}")
            else:
                logger.warning("どのチャンネルにも接続していません。")
        except AttributeError as e:
            logger.error(f"起動時の情報取得エラー: {e}")
        except Exception as e:
            logger.error(f"起動時の予期せぬエラー: {e}", exc_info=True)

        logger.info("Bot is ready!")

    async def event_message(self, message):
        if message.echo:
            return

        if message.content.startswith('!'):
            await self.handle_commands(message)
            return

        await self.process_message(message)

    async def process_message(self, message):
        """通常のチャットメッセージを処理し、必要であれば翻訳する"""
        user = message.author.name.lower()
        original_content = message.content.strip()
        channel_name = message.channel.name

        logger.debug(f"[{channel_name}] {user}: {original_content}")

        if user in IGNORE_USERS:
            logger.debug(f"無視ユーザーのためスキップ: {user}")
            return
        if any(line in original_content for line in IGNORE_LINES):
            logger.debug(f"無視行パターンのためスキップ: {original_content}")
            return

        if len(original_content) < MIN_TRANSLATION_LENGTH:
            logger.debug(f"短すぎるためスキップ ({len(original_content)} < {MIN_TRANSLATION_LENGTH}): {original_content}")
            return
        if len(original_content) > MAX_TRANSLATION_LENGTH:
            logger.info(f"長すぎるためスキップ ({len(original_content)} > {MAX_TRANSLATION_LENGTH}): {original_content}")
            return

        content_to_process, urls, mentions = remove_urls_mentions(original_content)

        if not content_to_process and (urls or mentions):
            logger.debug(f"URL/メンションのみのメッセージのためスキップ: {original_content}")
            return

        if any(word in content_to_process.lower() for word in IGNORE_WORDS):
            logger.debug(f"無視単語が含まれるためスキップ: {content_to_process}")
            return

        if content_to_process.lower() in COMMON_SLANGS:
            logger.debug(f"定型句のためスキップ: {content_to_process}")
            return

        if not self.translation_enabled:
            logger.debug("翻訳機能が無効のためスキップ")
            return

        detected_lang, confidence = detect_language_with_confidence(content_to_process)

        if detected_lang is None and nlp:
            spacy_lang = detect_language_spacy(content_to_process)
            if spacy_lang:
                detected_lang = spacy_lang
                confidence = 0.9
                logger.debug(f"spaCyにより言語を決定: {detected_lang}")

        should_translate = False
        if detected_lang is None:
            logger.debug(f"言語不明のため翻訳試行: {content_to_process}")
            should_translate = True
        elif detected_lang == 'ja' and not contains_significant_japanese(content_to_process, JAPANESE_CHARACTER_THRESHOLD):
            logger.debug(f"日本語と判定されたが日本語文字が少ないため、翻訳試行: {content_to_process}")
            should_translate = True
        elif confidence < LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
            logger.debug(f"言語検出の信頼度が低いため ({confidence:.2f} < {LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD})、翻訳試行: {content_to_process}")
            should_translate = True
        else:
            logger.debug(f"翻訳対象言語 ({detected_lang}, conf={confidence:.2f}): {content_to_process}")
            should_translate = True

        if should_translate:
            try:
                translation_result = await translate(content_to_process, 'ja')

                if translation_result:
                    translated_text, source_lang = translation_result
                    if translated_text.strip() == content_to_process.strip():
                        logger.debug(f"翻訳結果が元テキストと同じため表示スキップ: {translated_text}")
                    else:
                        display_lang = source_lang if source_lang != "unknown" else detected_lang if detected_lang else "?"
                        await message.channel.send(f"{user} ({display_lang.upper()}): {translated_text}")
                        logger.info(f"翻訳表示: [{channel_name}] {user} ({display_lang.upper()}) -> {translated_text}")
                else:
                    logger.warning(f"翻訳に失敗しました (APIエラー等): {content_to_process}")

            except Exception as e:
                logger.error(f"メッセージ処理中に予期せぬエラーが発生しました: {e}", exc_info=True)

    # --- コマンド ---
    @commands.command(name='ping')
    async def ping(self, ctx: commands.Context):
        await ctx.send(f'Pong! ({round(self.latency * 1000)}ms)')

    @commands.command(name='togglet')
    async def toggle_translation(self, ctx: commands.Context):
        """自動翻訳の有効/無効を切り替えます。"""
        if not (ctx.author.is_mod or ctx.author.name.lower() == CHANNEL.lower()):
            await ctx.send("このコマンドは配信者またはモデレーターのみ使用できます。")
            return

        self.translation_enabled = not self.translation_enabled
        status = "有効" if self.translation_enabled else "無効"
        await ctx.send(f"{ctx.author.name}、自動翻訳を{status}にしました。")
        logger.info(f"自動翻訳が {status} に変更されました by {ctx.author.name}")

    @commands.command(name='jatoen')
    async def translate_ja_to_en(self, ctx: commands.Context, *, text: str):
        """日本語テキストを英語に翻訳します。"""
        try:
            cleaned_text, _, _ = remove_urls_mentions(text)
            if not cleaned_text:
                await ctx.send("翻訳するテキストがありません。")
                return
            translation_result = await translate(cleaned_text, 'en')
            if translation_result:
                translated_text, source_lang = translation_result
                await ctx.send(f"{ctx.author.name} (JA->EN): {translated_text}")
            else:
                await ctx.send("翻訳に失敗しました。")
        except Exception as e:
            logger.error(f"jatoen コマンドエラー: {e}", exc_info=True)
            await ctx.send(f"翻訳エラーが発生しました: {e}")

    @commands.command(name='clear_cache')
    async def clear_cache(self, ctx: commands.Context):
        """翻訳キャッシュをクリアします。"""
        if not (ctx.author.is_mod or ctx.author.name.lower() == CHANNEL.lower()):
            await ctx.send("このコマンドは配信者またはモデレーターのみ使用できます。")
            return
        global translation_cache
        cache_size_before = len(translation_cache.cache)
        translation_cache = LRUCache(TRANSLATION_CACHE_SIZE)
        await ctx.send(f'{ctx.author.name}、翻訳キャッシュをクリアしました。(クリア前: {cache_size_before} 件)')
        logger.info(f"翻訳キャッシュがクリアされました by {ctx.author.name}")

    @commands.command(name='translate', aliases=['tr'])
    async def translate_to_lang(self, ctx: commands.Context, lang_code: str, *, text: str):
        """指定した言語コードにテキストを翻訳します。"""
        target_lang = lang_code.lower()
        if target_lang not in SUPPORTED_LANGUAGES and target_lang != 'ja':
            await ctx.send(f"サポートされていない言語コードです: {lang_code}。利用可能なコードは `!languages` で確認してください。")
            return

        try:
            cleaned_text, _, _ = remove_urls_mentions(text)
            if not cleaned_text:
                await ctx.send("翻訳するテキストがありません。")
                return
            translation_result = await translate(cleaned_text, target_lang)
            if translation_result:
                translated_text, source_lang = translation_result
                await ctx.send(f"{ctx.author.name} ({source_lang.upper()}->{target_lang.upper()}): {translated_text}")
            else:
                await ctx.send("翻訳に失敗しました (APIエラー等の可能性)。")

        except Exception as e:
            logger.error(f"translate コマンドエラー: {e}", exc_info=True)
            await ctx.send(f"翻訳エラーが発生しました: {e}")

    @commands.command(name='languages')
    async def list_languages(self, ctx: commands.Context):
        """利用可能な言語コード一覧を表示します。"""
        try:
            lang_items = [f"{code}: {name}" for code, name in SUPPORTED_LANGUAGES.items()]
            max_chars_per_message = 450
            current_message = "利用可能な言語コード一覧:\n"
            for item in lang_items:
                if len(current_message) + len(item) + 1 > max_chars_per_message:
                    await ctx.send(current_message)
                    current_message = item + "\n"
                else:
                    current_message += item + "\n"
            if current_message:
                await ctx.send(current_message)

        except Exception as e:
            logger.exception(f"言語リスト表示エラー: {e}")
            await ctx.send(f"言語コード一覧の表示に失敗しました: {e}")

if __name__ == "__main__":
    if not os.path.exists("config.py"):
        logger.warning("設定ファイル config.py が見つかりません。デフォルト設定で動作します。")
    if nlp is None:
        logger.warning("spaCy日本語モデルがロードされていないため、spaCyによる言語検出は使用されません。")

    bot = Bot()
    try:
        bot.run()
    except Exception as e:
        logger.critical(f"Botの実行中に致命的なエラーが発生しました: {e}", exc_info=True)
