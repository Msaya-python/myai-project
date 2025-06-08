import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 📁 ベースディレクトリとログパス
base_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(base_dir, "logs", "emotion_log.csv")

# 🎯 感情スコア変換マップ
emotion_to_score = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}

# 🖋 フォント設定（NotoSansに固定）
noto_path = os.path.join(base_dir, "NOTOSANSJP-VF.TTF")
if os.path.exists(noto_path):
    font_prop = font_manager.FontProperties(fname=noto_path)
else:
    print("⚠ フォントが見つかりませんでした。デフォルトフォントを使用します。")
    font_prop = None

# 📥 ログ読み込み
def load_log():
    if not os.path.exists(log_file):
        print("⚠ ログファイルが見つかりません。")
        return None

    records = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) == 3:
                try:
                    timestamp = pd.to_datetime(parts[0], format="%Y-%m-%d %H:%M:%S")
                    records.append({"datetime": timestamp, "text": parts[1], "emotion": parts[2]})
                except:
                    continue
            elif len(parts) == 4:
                try:
                    timestamp = pd.to_datetime(parts[0] + " " + parts[1], format="%Y-%m-%d %H:%M:%S")
                    records.append({"datetime": timestamp, "text": parts[2], "emotion": parts[3]})
                except:
                    continue

    return pd.DataFrame(records) if records else None

# 📊 折れ線グラフ
def plot_emotion_line(df):
    df["score"] = df["emotion"].map(emotion_to_score)
    plt.figure(figsize=(10, 4))
    plt.plot(df["datetime"], df["score"], marker="o", linestyle="-", color="blue")
    plt.yticks([-1, 0, 1], ["negative", "neutral", "positive"], fontproperties=font_prop)
    plt.title("感情の変化（折れ線グラフ）", fontproperties=font_prop)
    plt.xlabel("時刻", fontproperties=font_prop)
    plt.ylabel("感情スコア", fontproperties=font_prop)
    plt.xticks(rotation=45, fontproperties=font_prop)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 📈 移動平均グラフ
def plot_emotion_moving_average(df, window_size=5):
    df["score"] = df["emotion"].map(emotion_to_score)
    df["moving_avg"] = df["score"].rolling(window=window_size).mean()
    plt.figure(figsize=(12, 4))
    plt.plot(df["datetime"], df["score"], label="感情スコア", color="blue", alpha=0.5)
    plt.plot(df["datetime"], df["moving_avg"], label=f"{window_size}件移動平均", color="orange", linewidth=2)
    plt.yticks([-1, 0, 1], ["negative", "neutral", "positive"], fontproperties=font_prop)
    plt.title("感情の変動と移動平均", fontproperties=font_prop)
    plt.xlabel("時刻", fontproperties=font_prop)
    plt.ylabel("スコア", fontproperties=font_prop)
    plt.xticks(rotation=45, fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 📊 棒グラフ
def plot_emotion_bar(df):
    counts = df["emotion"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["green", "gray", "red"])
    plt.title("感情の出現回数（棒グラフ）", fontproperties=font_prop)
    plt.xlabel("感情", fontproperties=font_prop)
    plt.ylabel("回数", fontproperties=font_prop)
    plt.xticks(rotation=0, fontproperties=font_prop)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

# 🧠 メイン処理
def main():
    df = load_log()
    if df is not None and not df.empty:
        plot_emotion_line(df)
        plot_emotion_moving_average(df, window_size=5)
        plot_emotion_bar(df)
    else:
        print("⚠ ログが空か、読み込めませんでした。")

if __name__ == "__main__":
    main()

