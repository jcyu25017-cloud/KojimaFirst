import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

class Agent:
    def __init__(self, name, alpha, base_thresh):
        self.name = name
        # 個性を定義：alpha(欲求増加率), base_thresh(我慢強さ)
        self.alpha = alpha
        self.base_thresh = base_thresh
        
        self.drive = 0.0
        self.emotion = np.array([0.0, 0.0]) # [覚醒度, 快・不快]
        self.weight = np.array([5.0, -2.0]) # 感情が閾値に与える重み
        
        # グラフ描画用の履歴
        self.history = {"drive": [], "thresh": [], "won": []}

    def update_internal_state(self):
        """内部状態（欲求と動的閾値）の更新"""
        # 感情のランダムな揺らぎ
        noise = np.random.uniform(-0.1, 0.1, 2)
        self.emotion = np.clip(self.emotion + noise, -1.0, 1.0)
        
        # 欲求の蓄積
        self.drive += self.alpha
        
        # 動的閾値の計算
        self.dynamic_thresh = self.base_thresh - np.dot(self.weight, self.emotion)
        
        # 履歴の保存
        self.history["drive"].append(self.drive)
        self.history["thresh"].append(self.dynamic_thresh)

    def calculate_bid(self):
        """入札額（効用）の計算"""
        # 欲求が閾値を超えている分だけ、強く要求する（マイナスの場合は0）
        bid_value = max(0.0, self.drive - self.dynamic_thresh)
        return bid_value

    def consume_resource(self, step):
        """リソースを獲得し、欲求を満たす"""
        self.drive = 0.0
        self.history["won"].append((step, self.drive))
        print(f"    🎉 {self.name} がリソースを獲得しました！(欲求リセット)")

class Environment:
    def __init__(self):
        # 3つの異なる個性を持つエージェントを生成
        self.agents = [
            Agent("Agent_A (Gourmet)", alpha=3.0, base_thresh=20.0), # 欲求が高まりやすい
            Agent("Agent_B (Normal)",  alpha=2.0, base_thresh=25.0), # 標準的
            Agent("Agent_C (Stoic)",   alpha=1.0, base_thresh=35.0)  # 我慢強い
        ]
        self.time_steps = []

    def run_simulation(self, steps):
        for step in range(1, steps + 1):
            self.time_steps.append(step)
            print(f"\n--- Step {step} ---")
            
            # 1. 全エージェントの内部状態を更新
            for agent in self.agents:
                agent.update_internal_state()
            
            # 2. 環境にリソースが出現したか判定（ここでは3ステップごとに1つのケーキが出現）
            if step % 3 == 0:
                print("  🍰 [環境] リソースが出現しました！入札を開始します。")
                
                bids = {}
                # 3. 各エージェントが入札を行う
                for agent in self.agents:
                    bid = agent.calculate_bid()
                    bids[agent] = bid
                    if bid > 0:
                        print(f"    ✋ {agent.name} の入札額: {bid:.2f}")
                
                # 4. 落札者の決定（最大入札額のエージェント）
                max_bid = max(bids.values())
                if max_bid > 0:
                    # 最大入札額を出したエージェントを抽出
                    winners = [agent for agent, bid in bids.items() if bid == max_bid]
                    winner = winners[0] # 同額の場合はリストの先頭が勝利
                    winner.consume_resource(step - 1) # 配列インデックス調整のため-1
                else:
                    print("    ...誰も入札しませんでした。（全員欲求が閾値未満）")

    def plot_results(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Multi-Agent Negotiation: Resource Allocation", fontsize=14)

        for i, agent in enumerate(self.agents):
            ax = axes[i]
            # 欲求と閾値のプロット
            ax.plot(self.time_steps, agent.history["drive"], label="Drive (Appetite)", color="blue")
            ax.plot(self.time_steps, agent.history["thresh"], label="Dynamic Threshold", color="red", linestyle="--")
            
            # 勝利（リソース獲得）ポイントのプロット
            if agent.history["won"]:
                won_steps, won_vals = zip(*agent.history["won"])
                ax.scatter(won_steps, won_vals, color="gold", s=200, marker="*", zorder=5, label="Resource Won")
            
            ax.set_title(agent.name)
            ax.set_ylabel("Level")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    env = Environment()
    print("マルチエージェント自動交渉シミュレーションを開始します...")
    # 50ステップ実行
    env.run_simulation(50)
    print("\nシミュレーション完了。結果をグラフ化します...")
    env.plot_results()