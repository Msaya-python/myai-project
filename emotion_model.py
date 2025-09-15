from transformers import pipeline

# 感情分類モデル（Hugging Faceの日本語モデル）
classifier = pipeline(
    "text-classification",
    model="jarvisx17/japanese-sentiment-analysis",
    framework="pt"
)

def classify_emotion(text):
    """
    テキストを分類し、positive / neutral / negative を返す
    """
    result = classifier(text)
    if result and isinstance(result, list):
        label = result[0]['label'].lower()
        return label
    return "neutral"
