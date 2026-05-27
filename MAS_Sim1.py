import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # ← これを追加します。必ず pyplot のインポート前に書いてください。
import matplotlib.pyplot as plt
import time

# ※グラフの日本語文字化けを防ぐため、ラベル等は英語/ローマ字表記にしています
# 日本語を使いたい場合は pip install japanize-matplotlib を導入してください

class AgentMind:
    def __init__(self):
        self.emotion = np.array([0.8, 0.1]) 
        
        self.drives = {
            "Appetite": {"value": 0.0, "alpha": 2.5, "base_thresh": 20.0, "weight": np.array([5.0, -2.0])},
            "Sleep": {"value": 0.0, "alpha": 1.5, "base_thresh": 15.0, "weight": np.array([-4.0, 3.0])},
            "Learning": {"value": 0.0, "alpha": 1.0, "base_thresh": 25.0, "weight": np.array([8.0, 5.0])}
        }
        
        # グラフ描画用のデータ履歴を保存する辞書
        self.history = {
            "time": [],
            "drives": {name: {"value": [], "thresh": [], "fires": []} for name in self.drives.keys()}
        }

    def update_step(self, step):
        print(f"--- Step {step} ---")
        
        noise = np.random.uniform(-0.1, 0.1, 2)
        self.emotion = np.clip(self.emotion + noise, -1.0, 1.0)
        
        self.history["time"].append(step)

        for name, drive in self.drives.items():
            drive["value"] += drive["alpha"]
            dynamic_thresh = drive["base_thresh"] - np.dot(drive["weight"], self.emotion)
            
            # 履歴の記録
            self.history["drives"][name]["thresh"].append(dynamic_thresh)
            
            if drive["value"] >= dynamic_thresh:
                print(f"  >>> 💡 Action Fired: {name}")
                # 発火直前の値を記録（グラフの頂点用）
                self.history["drives"][name]["value"].append(drive["value"])
                self.history["drives"][name]["fires"].append((step, drive["value"]))
                
                drive["value"] = 0.0  # リセット
            else:
                self.history["drives"][name]["value"].append(drive["value"])

    def plot_results(self):
        """シミュレーション結果をグラフ化するメソッド"""
        # 欲求の数だけ縦にグラフを並べる設定
        num_drives = len(self.drives)
        fig, axes = plt.subplots(num_drives, 1, figsize=(10, 3 * num_drives), sharex=True)
        fig.suptitle("Agent Mind Simulation: Drives vs Dynamic Thresholds", fontsize=14)

        for ax, (name, data) in zip(axes, self.history["drives"].items()):
            time_steps = self.history["time"]
            
            # 欲求値の推移（青い実線）
            ax.plot(time_steps, data["value"], label=f"{name} Drive", color="blue", linewidth=2)
            
            # 動的閾値の推移（赤い点線）
            ax.plot(time_steps, data["thresh"], label=f"Dynamic Threshold", color="red", linestyle="--")
            
            # 発火ポイントのマーカー（黄色い星マーク）
            if data["fires"]:
                fire_times, fire_vals = zip(*data["fires"])
                ax.scatter(fire_times, fire_vals, color="gold", s=150, marker="*", zorder=5, label="Action Fired")

            ax.set_ylabel("Level")
            ax.set_title(f"[{name}]")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.show()

# --- シミュレーションの実行 ---
if __name__ == "__main__":
    mind = AgentMind()
    
    # 50ステップ分の時間経過をシミュレーション（グラフの変化を見るため回数を増やしています）
    for i in range(1, 51):
        mind.update_step(i)
        # グラフ描画時は高速に回すためスリープを省略、または極短くします
        # time.sleep(0.05) 
        
    # 最後にグラフを表示
    print("\nシミュレーション完了。グラフを描画します...")
    mind.plot_results()