# emotion_graph.py
# 時系列＋クラス別カウント。日本語フォント・色分け・同時表示＆保存対応

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib.gridspec import GridSpec

# === 設定から既定パス ===
try:
    import config
    DEFAULT_CSV = config.LOG_FILE_PATH
except Exception:
    DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "sora_emotion_log.csv")

# === 日本語フォント ===
def _set_jp_font():
    cands = ["Meiryo","Yu Gothic UI","MS Gothic","Hiragino Sans","Hiragino Kaku Gothic ProN",
             "Noto Sans CJK JP","IPAexGothic","TakaoGothic"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in cands if c in avail), cands[0])
    rcParams["font.family"] = chosen
    rcParams["axes.unicode_minus"] = False
    return chosen

# === CSV読込（堅牢） ===
def _read_csv(path: str) -> pd.DataFrame:
    tried = []
    for enc in ("utf-8-sig","utf-8","cp932"):
        try:
            return pd.read_csv(path, header=None, names=["date","time","text","emotion"], encoding=enc)
        except Exception as e:
            tried.append(f"{enc}:{e}")
    raise RuntimeError("CSV読み込みに失敗（試したenc: " + " | ".join(tried) + ")")

def load_emotion(primary: str):
    cands = [primary]
    if primary != DEFAULT_CSV:
        cands.append(DEFAULT_CSV)
    old = os.path.join(os.path.dirname(DEFAULT_CSV), "emotion_log.csv")
    if old not in cands:
        cands.append(old)

    for p in cands:
        if os.path.exists(p):
            df = _read_csv(p)
            break
    else:
        raise FileNotFoundError(f"CSVが見つかりません: {primary}（候補: {cands}）")

    df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str), errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    emo_map = {"positive":1, "neutral":0, "negative":-1}
    df["score"] = df["emotion"].map(emo_map)
    df = df.dropna(subset=["score"])
    if df.empty:
        raise ValueError("有効データがありません。")
    return df, p

def plot_all(df: pd.DataFrame, out_path="emotion_graph.png", show=True,
             rolling=0, resample_rule=None, figw=12, figh=8):
    # 配色
    color_map = {"positive":"#2ca02c", "neutral":"#7f7f7f", "negative":"#d62728"}
    line_color = "#1f77b4"

    # カウント
    counts = df["emotion"].value_counts().reindex(["positive","neutral","negative"], fill_value=0)
    total = int(counts.sum())

    # ===== レイアウト：constrained_layout で重なり回避 =====
    fig = plt.figure(figsize=(figw, figh), constrained_layout=True)
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[2.3, 0.22, 1.0])  # 真ん中は“スペーサー”

    # ===== 上：時系列 =====
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)

    plot_df = df.copy()
    if resample_rule:
        plot_df = (
            plot_df.set_index("datetime")["score"]
            .resample(resample_rule).mean()
            .to_frame("score").reset_index()
        )

    ax.plot(plot_df["datetime"], plot_df["score"], color=line_color, linewidth=1.4, alpha=0.9, label="感情スコア")

    # 各点を色分け（生ログ基準）
    for emo, c in color_map.items():
        sub = df[df["emotion"]==emo]
        if not sub.empty:
            ax.scatter(sub["datetime"], sub["score"], s=28, color=c, label=emo, zorder=3)

    # 移動平均
    if rolling and rolling > 1:
        rm = plot_df["score"].rolling(rolling, min_periods=max(1, rolling//2)).mean()
        ax.plot(plot_df["datetime"], rm, linestyle="--", linewidth=2.0, color="#000000", alpha=0.6,
                label=f"移動平均({rolling})")

    # 平均ライン
    avg = float(plot_df["score"].mean())
    ax.axhline(avg, linestyle=":", linewidth=1.6, color="#000000", alpha=0.5, label=f"平均={avg:.2f}")

    ax.set_title("ソラ感情ログ", fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("日時", fontsize=12, labelpad=10)     # ← 下方向の余白を広めに
    ax.set_ylabel("感情", fontsize=12)
    ax.set_yticks([-1,0,1]); ax.set_yticklabels(["negative","neutral","positive"])
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(25); lbl.set_horizontalalignment("right")
    ax.tick_params(axis="x", pad=8)                    # ← 目盛ラベルと下段の干渉を回避
    ax.legend(loc="upper left", ncol=2)

    # 内訳テキスト
    ax.text(0.99, 0.02,
            f"Total: {total}\nPos: {counts['positive']}  Neu: {counts['neutral']}  Neg: {counts['negative']}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    # ===== 下：棒グラフ =====
    axb = fig.add_subplot(gs[2, 0])
    bars = axb.bar(["positive","neutral","negative"],
                   [counts["positive"], counts["neutral"], counts["negative"]],
                   color=[color_map["positive"], color_map["neutral"], color_map["negative"]])
    axb.set_ylabel("件数")
    axb.set_title("感情カウント", fontsize=13, pad=6)
    axb.grid(True, axis="y", alpha=0.2)
    for b in bars:
        axb.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, str(int(b.get_height())),
                 ha="center", va="bottom", fontsize=10)

    # ===== 保存＆表示 =====
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"✅ 画像を保存しました: {out_path}")
    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="MYAI 感情ログ可視化（時系列＋カウント）")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="感情ログCSVのパス")
    parser.add_argument("--out", default="emotion_graph.png", help="出力画像パス")
    # デフォルトで表示。オフにしたいときだけ --no-show
    parser.add_argument("--show", dest="show", action="store_true", default=True, help="ウィンドウ表示を行う（既定: ON）")
    parser.add_argument("--no-show", dest="show", action="store_false", help="ウィンドウ表示を行わない")
    parser.add_argument("--rolling", type=int, default=0, help="移動平均の窓幅（例:7）")
    parser.add_argument("--resample", dest="resample_rule", default=None, help="再サンプル規則（D/W/Mなど）")
    parser.add_argument("--figw", type=float, default=12.0, help="図の横幅（インチ）")
    parser.add_argument("--figh", type=float, default=8.0, help="図の高さ（インチ）")
    args = parser.parse_args()

    chosen = _set_jp_font()
    print(f"ℹ️ 日本語フォント: {chosen}")
    print(f"ℹ️ 参照CSV: {args.csv}")

    try:
        df, used = load_emotion(args.csv)
        if used != args.csv:
            print(f"ℹ️ 実際に使用したCSV: {used}")
    except Exception as e:
        print(f"🛑 CSV読み込みエラー: {e}")
        sys.exit(1)

    try:
        plot_all(df,
                 out_path=args.out, show=args.show,
                 rolling=max(0, args.rolling),
                 resample_rule=args.resample_rule,
                 figw=args.figw, figh=args.figh)
    except Exception as e:
        print(f"🛑 プロットエラー: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
