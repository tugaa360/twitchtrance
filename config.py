# config.py

# --- 基本設定 ---
IGNORE_USERS = ["Nightbot", "StreamElements"] # 無視するユーザー名 (小文字)
IGNORE_LINES = ["!command", "http://"]      # この文字列を含む行は無視
IGNORE_WORDS = ["spam_word", "bad_word"]   # この単語を含むメッセージは無視 (小文字)

# --- 翻訳設定 ---
MAX_TRANSLATION_LENGTH = 500 # 翻訳するメッセージの最大長
MIN_TRANSLATION_LENGTH = 5   # これより短いメッセージは翻訳しない (定型句チェックの後)
LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.80 # 言語検出の最低信頼度 (これ未満でも翻訳試行するロジックに注意)
JAPANESE_CHARACTER_THRESHOLD = 0.3 # 日本語と判定されても、日本語文字の割合がこれ未満なら翻訳試行 (0.0-1.0)
COMMON_SLANGS = ["gg", "lol", "pog", "kappa", "lmao", "omegalul", "kekw", "rip", "f", "nice"] # 翻訳しない定型句 (小文字)
IGNORE_URLS = True           # メッセージ中のURLを翻訳対象から除くか
IGNORE_MENTIONS = True       # メッセージ中の @メンション を翻訳対象から除くか

# --- Twitch API (環境変数で設定推奨) ---
# TWITCH_CLIENT_ID = "your_client_id"
# TWITCH_ACCESS_TOKEN = "your_access_token"
# TWITCH_CHANNEL = "your_channel_name"

# --- キャッシュ設定 (環境変数で設定推奨) ---
# TRANSLATION_CACHE_SIZE = 100
