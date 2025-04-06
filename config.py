# config.py
import re

# --- 基本設定 ---
IGNORE_USERS = ["Nightbot", "StreamElements"] # 無視するユーザー名 (小文字)
IGNORE_LINES = ["!command", "http://"]      # この文字列を含む行は無視
IGNORE_WORDS = ["spam_word", "bad_word"]   # この単語を含むメッセージは無視 (小文字)

 # サポートする翻訳対象言語のリスト
SUPPORTED_LANGUAGES = ['ja', 'en', 'es', 'fr', 'de', 'ko', 'zh-cn'] # サポートする言語を追加
# --- 翻訳設定 ---
MAX_TRANSLATION_LENGTH = 500 # 翻訳するメッセージの最大長
MIN_TRANSLATION_LENGTH = 1  # これより短いメッセージは翻訳しない (定型句チェックの後)
LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.7# 言語検出の最低信頼度 (これ未満でも翻訳試行するロジックに注意)
JAPANESE_CHARACTER_THRESHOLD = 0.3 # 日本語と判定されても、日本語文字の割合がこれ未満なら翻訳試行 (0.0-1.0)
COMMON_SLANGS = ["gg", "lol", "pog", "kappa", "lmao", "omegalul", "kekw", "rip", "f", "nice"] # 翻訳しない定型句 (小文字)
IGNORE_URLS = True           # メッセージ中のURLを翻訳対象から除くか
IGNORE_MENTIONS = True       # メッセージ中の @メンション を翻訳対象から除くか
# 翻訳失敗時のメッセージを追加
TRANSLATION_FAILURE_MESSAGE = "翻訳に失敗しました。"

# --- 既存のコードの適切な箇所に追加する定数 ---
# Twitchの一般的な絵文字のリスト（必要に応じて拡張）
TWITCH_EMOTES = [
    "Kappa", "PogChamp", "BibleThump", "Kreygasm", "4Head", "SwiftRage", "TriHard", 
    "DansGame", "FrankerZ", "SMOrc", "Jebaited", "LUL", "KappaPride", "VoHiYo", 
    "CoolCat", "SabaPing", "TehePelo", "ThunBeast", "KonCha", "RaccAttack",
    "HeyGuys", "ResidentSleeper", "NotLikeThis", "WutFace", "BabyRage"
]

# 絵文字パターンの正規表現（絵文字に使われる可能性のある文字パターン）
EMOTE_PATTERN = r'[A-Z][a-z]*[A-Z][a-zA-Z]*'  # 例: PogChamp, KappaPride など

# --- 新しい関数を追加 ---
def contains_twitch_emote(text: str) -> bool:
    """
    メッセージにTwitch絵文字が含まれているかどうかを確認する
    
    1. 既知の絵文字リストとの完全一致
    2. メッセージ中の単語が既知の絵文字リストに含まれるか
    3. 絵文字パターンに一致する単語がないか
    
    いずれかの条件に当てはまる場合、True を返す
    """
    # メッセージ全体が絵文字だけの場合
    if text in TWITCH_EMOTES:
        return True
    
    # メッセージを単語に分割して各単語をチェック
    words = text.split()
    for word in words:
        if word in TWITCH_EMOTES:
            return True
    
    # 絵文字パターンに一致する単語がないかチェック
    matches = re.findall(EMOTE_PATTERN, text)
    if matches:
        for match in matches:
            # パターンに一致する単語がTwitch絵文字っぽい場合
            if len(match) >= 4 and match[0].isupper():
                return True
    
    return False

# --- 既存の should_translate_message 関数を修正 ---
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
    # Twitch絵文字チェックを追加
    if contains_twitch_emote(content):
        return False
    return True
# --- Twitch API (環境変数で設定推奨) ---
# TWITCH_CLIENT_ID = "your_client_id"
# TWITCH_ACCESS_TOKEN = "your_access_token"
# TWITCH_CHANNEL = "your_channel_name"

# --- キャッシュ設定 (環境変数で設定推奨) ---
# TRANSLATION_CACHE_SIZE = 100
