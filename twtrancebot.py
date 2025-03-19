import logging
import os
import asyncio
import json
import re
from collections import OrderedDict
import unicodedata
from typing import Optional, Tuple, Dict

from twitchio.ext import commands
from googletrans import Translator
import spacy
# from spellchecker import SpellChecker
import langdetect
from dotenv import load_dotenv
import urllib.request
import time

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .envファイルの読み込み
load_dotenv()

# 環境変数から設定を読み込み
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN")
CHANNEL = os.getenv("TWITCH_CHANNEL")
TRANSLATION_CACHE_SIZE = os.getenv("TRANSLATION_CACHE_SIZE", "100")
TRANSLATION_RETRY_COUNT = int(os.getenv("TRANSLATION_RETRY_COUNT", "3"))  # リトライ回数のデフォルト値
TRANSLATION_RETRY_DELAY = int(os.getenv("TRANSLATION_RETRY_DELAY", "1"))  # リトライ間隔のデフォルト値

print(f"Client ID: {CLIENT_ID}")
print(f"Access Token: {ACCESS_TOKEN}")
print(f"Channel: {CHANNEL}")

if not CLIENT_ID:
    raise ValueError("環境変数TWITCH_CLIENT_IDが設定されていません")
if not ACCESS_TOKEN:
    raise ValueError("環境変数TWITCH_TOKENが設定されていません")
if not CHANNEL:
    raise ValueError("環境変数TWITCH_CHANNELが設定されていません")

# TRANSLATION_CACHE_SIZEが整数であることを確認
try:
    TRANSLATION_CACHE_SIZE = int(TRANSLATION_CACHE_SIZE)
except ValueError:
    logging.error("環境変数TRANSLATION_CACHE_SIZEは整数である必要があります。デフォルト値100を使用します。")
    TRANSLATION_CACHE_SIZE = 100

# 翻訳リトライ回数と間隔の確認
try:
    TRANSLATION_RETRY_COUNT = int(TRANSLATION_RETRY_COUNT)
    if TRANSLATION_RETRY_COUNT < 0:
        raise ValueError("TRANSLATION_RETRY_COUNTは0以上の整数である必要があります")
except ValueError:
    logging.error(
        "環境変数TRANSLATION_RETRY_COUNTは0以上の整数である必要があります。デフォルト値3を使用します。"
    )
    TRANSLATION_RETRY_COUNT = 3

try:
    TRANSLATION_RETRY_DELAY = int(TRANSLATION_RETRY_DELAY)
    if TRANSLATION_RETRY_DELAY < 0:
        raise ValueError("TRANSLATION_RETRY_DELAYは0以上の整数である必要があります")
except ValueError:
    logging.error(
        "環境変数TRANSLATION_RETRY_DELAYは0以上の整数である必要があります。デフォルト値1を使用します。"
    )
    TRANSLATION_RETRY_DELAY = 1

# 翻訳機能の初期化
try:
    TRANSLATOR = Translator()
except Exception as e:
    logging.error(f"Translatorの初期化に失敗しました: {e}", exc_info=True)
    TRANSLATOR = None

# NLPモデルの初期化
try:
    nlp = spacy.load("ja_core_news_sm")
except OSError:
    print("spaCyの日本語モデルをダウンロード中...")
    spacy.cli.download("ja_core_news_sm")
    nlp = spacy.load("ja_core_news_sm")

# スペルチェッカーの初期化
spell = None


class LRUCache:
    """
    LRU (Least Recently Used) キャッシュクラス。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def put(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value


# fasttextモデルのロード
USE_FASTTEXT = False
try:
    import fasttext

    try:
        ft_model = fasttext.load_model('lid.176.bin')
        USE_FASTTEXT = True
    except ValueError:
        print("fasttextモデルをダウンロード中...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, 'lid.176.bin')
        ft_model = fasttext.load_model('lid.176.bin')
        USE_FASTTEXT = True
except ModuleNotFoundError:
    print("fasttextモジュールが見つかりませんでした。fasttextを使用しません。")
    ft_model = None


def detect_language_with_confidence(text, min_prob=0.5):
    """
    テキストの言語を検出し、信頼度がmin_prob以上の場合のみ結果を返す。
    """
    try:
        langs = langdetect.detect_langs(text)
        for lang in langs:
            if lang.prob > min_prob:
                return lang.lang
        return None
    except:
        return None


def detect_language_ensemble(text):
    """
    langdetectとfasttextの結果を組み合わせて言語を検出する。
    """
    try:
        langdetect_result = detect_language_with_confidence(text)
    except:
        langdetect_result = None

    if USE_FASTTEXT:
        try:
            fasttext_result = ft_model.predict(text)[0][0].replace('__label__', '')
        except:
            fasttext_result = None

        if langdetect_result == fasttext_result:
            return langdetect_result
        elif langdetect_result and fasttext_result:
            return langdetect_result
        else:
            return langdetect_result or fasttext_result
    else:
        return langdetect_result


def detect_language_safe(text):
    """
    言語検出を行い、エラーが発生した場合はNoneを返す。
    """
    try:
        return langdetect.detect(text)
    except langdetect.LangDetectException:
        return None


class Bot(commands.Bot):
    def __init__(self):
        super().__init__(token=ACCESS_TOKEN, prefix='!', initial_channels=[CHANNEL])
        self.translation_cache = LRUCache(capacity=TRANSLATION_CACHE_SIZE)
        self.source_language = None
        self.translation_enabled = True
        self.translation_retry_count = TRANSLATION_RETRY_COUNT
        self.translation_retry_delay = TRANSLATION_RETRY_DELAY
        self.languages = {  # 翻訳可能な言語一覧
            'en': '英語',
            'ja': '日本語',
            'ko': '韓国語',
            'zh-cn': '中国語 (簡体字)',
            'zh-tw': '中国語 (繁体字)',
            # 必要に応じて他の言語を追加
        }

    async def event_ready(self):
        print(f'準備完了 | {self.nick}')
        logging.info(f'準備完了 | {self.nick}')

    async def event_message(self, message):
        if message.echo:
            return
        if message.content.startswith('!'):
            await self.handle_commands(message)
        else:
            await self.process_message(message)

    def remove_html_tags(self, text: str) -> str:
        """HTMLタグを削除する"""
        return re.sub(r'<[^>]+>', '', text)

    def remove_control_characters(self, text: str) -> str:
        """制御文字を削除する"""
        return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

    def normalize_whitespace(self, text: str) -> str:
        """空白文字を正規化する"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_symbols(self, text: str) -> str:
        """記号を削除する"""
        return re.sub(r'[^\w\s]', '', text)

    def remove_urls_emails_phones(self, text: str) -> str:
        """URL、メールアドレス、電話番号を削除する"""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)
        return text

    def unicode_normalize(self, text: str) -> str:
        """Unicode正規化を行う"""
        return unicodedata.normalize('NFKC', text)

    def preprocess_text(self, text: str) -> str:
        """
        テキストの前処理を行う。
        """
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            text = self.remove_html_tags(text)
            text = self.remove_control_characters(text)
            text = self.normalize_whitespace(text)
            text = self.remove_symbols(text)
            text = self.remove_urls_emails_phones(text)
            text = self.unicode_normalize(text)
            text = self.normalize_slang(text)
            return text
        except Exception as e:
            logging.error(f"テキストの前処理中にエラーが発生しました: {e}", exc_info=True)
            return text

    def normalize_slang(self, text: str) -> str:
        """
        スラングを正規化する。
        """
        try:
            slang_dict = {
                "w": "笑",
                "草": "笑",
                "ktkr": "キタコレ",
            }
            for slang, normalized in slang_dict.items():
                text = text.replace(slang, normalized)
            return text
        except Exception as e:
            logging.error(f"スラングの正規化中にエラーが発生しました: {e}", exc_info=True)
            return text

    async def process_message(self, message):
        """
        受信したメッセージを処理し、翻訳してチャットに送信する。
        """
        content = message.content

        try:
            if not self.translation_enabled:
                return

            detected_language = detect_language_ensemble(content)
            if not detected_language:
                detected_language = 'ja'

            preprocessed_text = self.preprocess_text(content)

            translated_text, detected_language = await self.translate_message(
                preprocessed_text, source_language=self.source_language
            )

            if detected_language != 'ja':
                await message.channel.send(
                    f"[{detected_language}] {content} (日本語訳: {translated_text})"
                )

        except Exception as e:
            logging.exception(f"メッセージ処理エラー:", exc_info=True)
            await message.channel.send("メッセージの処理中にエラーが発生しました。")

    async def translate_message(
        self, text: str, target_language: str = 'ja', source_language: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        メッセージを翻訳し、翻訳されたテキストと検出された言語を返す。
        翻訳結果はキャッシュに保存される。
        """
        if TRANSLATOR is None:
            return text, 'ja'

        cache_key = (text, target_language, source_language)
        cached_result = self.translation_cache.get(cache_key)

        if cached_result:
            logging.info(f"翻訳キャッシュから取得: {text}")
            return cached_result['translated_text'], cached_result['detected_language']

        for attempt in range(self.translation_retry_count + 1):
            try:
                translation = TRANSLATOR.translate(
                    text, dest=target_language, src=source_language
                )
                translated_text = translation.text
                detected_language = translation.src
                self.translation_cache.put(
                    cache_key,
                    {'translated_text': translated_text, 'detected_language': detected_language},
                )
                return translated_text, detected_language
            except Exception as e:
                logging.error(
                    f"翻訳エラー (試行{attempt + 1}/{self.translation_retry_count + 1}): {e}",
                    exc_info=True,
                )
                if attempt < self.translation_retry_count:
                    await asyncio.sleep(self.translation_retry_delay)  # リトライ待機
                else:
                    # await message.channel.send(f"翻訳中にエラーが発生しました: {e}") #messageが定義されていないので削除
                    return "[翻訳エラー] " + text, 'ja'

    @commands.command(name='set_source_language')
    async def set_source_language(self, ctx: commands.Context, source_language: str):
        """
        翻訳元の言語を設定するコマンド。
        """
        if source_language not in self.languages:
            await ctx.send(
                f'{ctx.author.name}、無効な言語コードです。翻訳可能な言語コードは、!languages コマンドで確認してください。'
            )
            return
        self.source_language = source_language
        await ctx.send(f'{ctx.author.name}、翻訳元の言語を "{source_language}" に設定しました。')

    @commands.command(name='toggle_translation')
    async def toggle_translation(self, ctx: commands.Context):
        """
        翻訳の有効/無効を切り替えるコマンド。
        """
        self.translation_enabled = not self.translation_enabled
        status = "有効" if self.translation_enabled else "無効"
        await ctx.send(f'{ctx.author.name}、翻訳を{status}にしました。')

    @commands.command(name='tr')
    async def translate(self, ctx: commands.Context, target_language: str, *, text: str):
        """
        指定された言語に翻訳するコマンド。
        """
        if TRANSLATOR is None:
            await ctx.send("翻訳機能が利用できません。")
            return

        if target_language not in self.languages:
            await ctx.send(
                f'{ctx.author.name}、無効な言語コードです。翻訳可能な言語コードは、!languages コマンドで確認してください。'
            )
            return

        try:
            translated_text, detected_language = await self.translate_message(
                text, target_language=target_language
            )
            if detected_language != 'ja':
                await ctx.send(
                    f'{ctx.author.name}さんのメッセージ: "{text}" ({self.languages[target_language]}訳: "{translated_text}")'
                )
            else:
                await ctx.send(
                    f'{ctx.author.name}さんのメッセージ: "{text}" は日本語ではありません。'
                )
        except Exception as e:
            logging.error(
                f"{target_language}への翻訳中にエラーが発生しました: {e}", exc_info=True
            )
            await ctx.send(f'{ctx.author.name}、翻訳中にエラーが発生しました: {e}')

    @commands.command(name='languages')
    async def show_languages(self, ctx: commands.Context):
        """
        翻訳可能な言語一覧を表示するコマンド。
        """
        message = "翻訳可能な言語は以下の通りです:\n"
        for code, name in self.languages.items():
            message += f"{code}: {name}\n"
        await ctx.send(message.strip())

    @commands.command(name='set_retry')
    async def set_retry(self, ctx: commands.Context, count: int, delay: int):
        """
        翻訳リトライ回数と間隔を設定するコマンド。
        """
        if count < 0 or delay < 0:
            await ctx.send(
                f'{ctx.author.name}、リトライ回数と間隔は0以上の整数で指定してください。'
            )
            return
        self.translation_retry_count = count
        self.translation_retry_delay = delay
        await ctx.send(
            f'{ctx.author.name}、翻訳リトライを{self.translation_retry_count}回、間隔{self.translation_retry_delay}秒に設定しました。'
        )

bot = Bot()
bot.run()
