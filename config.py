try:
    from config_local import * # F401,F403
except ImportError:
    import os
    OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
    VOICEVOX_PORT       = int(os.getenv("VOICEVOX_PORT", "50021"))
    DEFAULT_SPEAKER_ID  = int(os.getenv("DEFAULT_SPEAKER_ID", "58"))
    LOG_FILE_PATH       = os.getenv("LOG_FILE_PATH", "./logs/sora_log.csv")
    VOICE_OUTPUT_PATH   = os.getenv("VOICE_OUTPUT_PATH", "./outputs/voice.wav")