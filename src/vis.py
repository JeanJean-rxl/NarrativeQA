import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 summary
df = pd.read_csv("summary_results.csv")

# 如果你之前没加，可以清理空格
df.columns = df.columns.str.strip()

# 可视化 fuzzy
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="scope_tokens",
    y="fuzzy_rate_80",
    hue="model",
    marker="o"
)

plt.title("Scope vs Fuzzy Score ≥ 0.8", fontsize=14)
plt.xlabel("Scope Size (Tokens: chunk_words × top_k)", fontsize=12)
plt.ylabel("Proportion of High-Quality Answers", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("scope_vs_fuzzy80.png")
plt.show()