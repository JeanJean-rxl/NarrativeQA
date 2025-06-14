import pandas as pd

# 读取你的 CSV 文件
df = pd.read_csv("data/rag_results_checkpoint-32748_chunk64_sample100.csv")

# 定义函数
def prediction_in_context(prediction: str, context: str) -> bool:
    pred = str(prediction).lower().strip()
    ctx = str(context).lower()
    return pred in ctx

# 应用函数
df["prediction_in_context"] = df.apply(
    lambda row: prediction_in_context(row["prediction"], row["retrieved_context"]),
    axis=1
)

# 保存或查看结果
df.to_csv("with_prediction_in_context_32748.csv", index=False)
print(df[["prediction", "prediction_in_context"]].head())