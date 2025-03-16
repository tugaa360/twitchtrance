import logging
import os
import asyncio
import json
import re
from collections import OrderedDict
import unicodedata

from twitchio.ext import commands
from googletrans import Translator
import spacy
# from spellchecker import SpellChecker  # indexerエラーのためコメントアウト
import langdetect
from dotenv import load_dotenv  # .envファイル読み込み用
import urllib.request

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .envファイルの読み込み
load_dotenv()

# 環境変数から設定を読み込み
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
ACCESS_TOKEN = os.getenv("TWITCH_ACCESS_TOKEN")
CHANNEL = os.getenv("TWITCH_CHANNEL")
TRANSLATION_CACHE_SIZE = os.getenv("TRANSLATION_CACHE_SIZE", "100")  # デフォルト値は100

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

# 翻訳機能の初期化
try:
    TRANSLATOR = Translator()
except Exception as e:
    logging.error(f"Translatorの初期化に失敗しました: {e}", exc_info=True)
    TRANSLATOR = None  # Translatorを使用不能にする

# NLPモデルの初期化
try:
    nlp = spacy.load("ja_core_news_sm")  # 日本語モデル
except OSError:
    print("spaCyの日本語モデルをダウンロード中...")
    spacy.cli.download("ja_core_news_sm")
    nlp = spacy.load("ja_core_news_sm")

# # スペルチェッカーの初期化 (indexerエラーのためコメントアウト)
# try:
#     spell = SpellChecker(language='ja')
# except ValueError as e:
#     print(f"スペルチェッカーの初期化に失敗しました: {e}")
#     print("日本語辞書ファイルがインストールされているか確認してください。")
#     spell = None
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
USE_FASTTEXT = False  # fasttextが利用可能かどうかを示すフラグ
try:
    import fasttext
    try:
        ft_model = fasttext.load_model('lid.176.bin')  # 176言語対応モデル
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
        return None  # 信頼できる言語が見つからなかった場合
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
            # 異なる結果の場合、langdetectの結果を優先
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
        self.translation_cache = LRUCache(capacity=TRANSLATION_CACHE_SIZE)  # LRUキャッシュ
        self.source_language = None  # 翻訳元の言語
        self.translation_enabled = True  # 翻訳の有効/無効を切り替えるフラグ

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

    # テキストの前処理
    def preprocess_text(self, text):
        """
        テキストの前処理を行う。
        """
        try:
            # 1. 文字エンコーディングの統一 (UTF-8に変換)
            text = text.encode('utf-8', errors='ignore').decode('utf-8')

            # 2. 不要な文字の削除 (HTMLタグ、制御文字など)
            text = re.sub(r'<[^>]+>', '', text)  # HTMLタグの削除
            text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')  # 制御文字の削除

            # 3. 空白文字の正規化
            text = re.sub(r'\s+', ' ', text)  # 連続する空白文字を1つに
            text = text.strip()  # 前後の空白を削除

            # 4. 大文字・小文字の統一 (必要に応じて)
            # text = text.lower()

            # 5. 記号の処理 (必要に応じて)
            text = re.sub(r'[^\w\s]', '', text)  # 記号の削除 (英数字、空白以外)

            # 6. URL、メールアドレス、電話番号などの処理 (必要に応じて)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLの削除
            text = re.sub(r'\S+@\S+', '', text)  # メールアドレスの削除
            text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)  # 電話番号の削除

            # 7. Unicode正規化
            text = unicodedata.normalize('NFKC', text)

            # 8. スラングの正規化
            text = self.normalize_slang(text)

            return text
        except Exception as e:
            logging.error(f"テキストの前処理中にエラーが発生しました: {e}", exc_info=True)
            return text  # エラーが発生した場合は、元のテキストを返す

    # スラングの正規化
    def normalize_slang(self, text):
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
            return text  # エラーが発生した場合は、元のテキストを返す

    async def process_message(self, message):
        """
        受信したメッセージを処理し、翻訳してチャットに送信する。
        """
        content = message.content

        try:
            # 翻訳が無効な場合は、メッセージを処理しない
            if not self.translation_enabled:
                return

            # 0. 言語の特定
            try:
                detected_language = detect_language_ensemble(content)
                if not detected_language:
                    detected_language = 'ja'  # 検出失敗時は日本語とする
            except Exception as e:
                logging.error(f"言語検出エラー: {e}", exc_info=True)
                detected_language = 'ja'  # 検出失敗時は日本語とする

            # 1. テキストの前処理
            preprocessed_text = self.preprocess_text(content)

            # 2. メッセージを翻訳
            try:
                translated_text, detected_language = await self.translate_message(preprocessed_text, source_language=self.source_language)
            except Exception as e:
                logging.error(f"翻訳中にエラーが発生しました: {e}", exc_info=True)
                await message.channel.send("翻訳中にエラーが発生しました。")
                return

            # 3. 翻訳されたメッセージをチャットに表示
            if detected_language != 'ja':
                await message.channel.send(f"[{detected_language}] {content} (日本語訳: {translated_text})")

        except Exception as e:
            logging.exception(f"メッセージ処理エラー:", exc_info=True)
            await message.channel.send("メッセージの処理中にエラーが発生しました。")

    async def translate_message(self, text, target_language='ja', source_language=None):
        """
        メッセージを翻訳し、翻訳されたテキストと検出された言語を返す。
        翻訳結果はキャッシュに保存される。
        """
        # Translatorが初期化されていない場合は、翻訳をスキップ
        if TRANSLATOR is None:
            return text, 'ja'  # 元のテキストと日本語を返す

        cache_key = (text, target_language, source_language)
        cached_result = self.translation_cache.get(cache_key)

        if cached_result:
            logging.info(f"翻訳キャッシュから取得: {text}")
            return cached_result['translated_text'], cached_result['detected_language']

        try:
            translation = TRANSLATOR.translate(text, dest=target_language, src=source_language)
            translated_text = translation.text
            detected_language = translation.src
            self.translation_cache.put(cache_key, {'translated_text': translated_text, 'detected_language': detected_language})
            return translated_text, detected_language
        except Exception as e:
            logging.error(f"翻訳エラー: {e}", exc_info=True)
            await message.channel.send(f"翻訳中にエラーが発生しました: {e}")  # ユーザーにエラーを通知
            return "[翻訳エラー] " + text, 'ja'  # エラーが発生した場合は、元のテキストと日本語を返す

    @commands.command(name='set_source_language')
    async def set_source_language(self, ctx: commands.Context, source_language: str):
        """
        翻訳元の言語を設定するコマンド。
        """
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

bot = Bot()
bot.run()