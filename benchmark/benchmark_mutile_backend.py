import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import json
import warnings

# 忽略 Seaborn 的一些 FutureWarning
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Liger Kernel Benchmark Visualizer")
    
    # 允许传入多个 CSV 文件
    parser.add_argument(
        "--csv-paths", 
        nargs="+", 
        default=["data/all_benchmark_data.csv"],
        help="Path(s) to the benchmark CSV files. Can provide multiple files (e.g., gpu.csv npu.csv)."
    )
    
    parser.add_argument("--kernel-name", type=str, required=True, help="Name of the kernel (e.g., swiglu, rope)")
    parser.add_argument("--metric-name", type=str, default="speed", help="Metric to visualize (speed or memory)")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Directory to save plots")
    
    return parser.parse_args()

def load_and_merge_data(csv_paths):
    dfs = []
    for path in csv_paths:
        if os.path.exists(path):
            try:
                # 读取 CSV
                df = pd.read_csv(path)
                # 简单清洗：去除空行或错误行
                df = df.dropna(subset=['kernel_name', 'y_value_50'])
                dfs.append(df)
                print(f"Successfully loaded: {path} ({len(df)} rows)")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Warning: File not found: {path}")
    
    if not dfs:
        raise ValueError("No valid data loaded.")
    
    return pd.concat(dfs, ignore_index=True)

def generate_plot_label(row):
    """生成图例标签：Provider + Device Name"""
    provider = row['kernel_provider']
    
    # 简化显卡名称，太长了图例不好看
    gpu = str(row['gpu_name'])
    if "NVIDIA" in gpu:
        gpu = gpu.replace("NVIDIA ", "")
    if "GeForce" in gpu:
        gpu = gpu.replace("GeForce ", "")
    if "Ascend" in gpu:
        # 华为 Ascend 有时候名字很长，取前缀
        gpu = gpu.split("_")[0] 
        
    return f"{provider} ({gpu})"

def plot_benchmark(df, kernel_name, metric_name, mode, output_dir):
    # 1. 筛选数据
    subset = df[
        (df["kernel_name"] == kernel_name) & 
        (df["metric_name"] == metric_name) & 
        (df["kernel_operation_mode"] == mode)
    ].copy()

    if subset.empty:
        print(f"No data found for {kernel_name} - {metric_name} - {mode}. Skipping.")
        return

    # 2. 构造组合标签 (Liger on A100 vs HF on Ascend)
    subset["legend_label"] = subset.apply(generate_plot_label, axis=1)
    
    # 3. 确保数值类型正确
    subset["x_value"] = pd.to_numeric(subset["x_value"])
    subset["y_value_50"] = pd.to_numeric(subset["y_value_50"])
    
    # 4. 获取单位和标签
    unit = subset["metric_unit"].iloc[0] if "metric_unit" in subset else "N/A"
    xlabel = subset["x_label"].iloc[0] if "x_label" in subset else "Input Size"
    
    # 排序，为了画线顺畅
    subset = subset.sort_values(by=["legend_label", "x_value"])

    # 5. 设置绘图风格
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # 定义线型：HuggingFace 用虚线，Liger 用实线 (可选优化)
    markers = {"liger": "o", "huggingface": "X", "torch": "^", "triton": "s"}
    
    # 绘制主图
    ax = sns.lineplot(
        data=subset,
        x="x_value",
        y="y_value_50",
        hue="legend_label",
        style="legend_label", # 让不同设备也有不同的线型/标记
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=9
    )

    # 6. 处理对数坐标 (如果数据跨度 > 100倍)
    y_min, y_max = subset["y_value_50"].min(), subset["y_value_50"].max()
    if y_max > 0 and y_min > 0 and (y_max / y_min > 100):
        ax.set_yscale("log")
        plt.ylabel(f"{metric_name} ({unit}) - Log Scale")
    else:
        plt.ylabel(f"{metric_name} ({unit})")

    # 7. 手动绘制误差棒 (Error Bars)
    # 因为 CSV 里已经有了 20/80 分位数值，直接用它们画误差
    colors = {line.get_label(): line.get_color() for line in ax.get_lines()}
    
    for label, group in subset.groupby("legend_label"):
        if label in colors:
            color = colors[label]
            # 计算误差距离
            lower_error = group["y_value_50"] - group["y_value_20"]
            upper_error = group["y_value_80"] - group["y_value_50"]
            
            # 防止负数误差导致绘图错误
            lower_error = lower_error.clip(lower=0)
            upper_error = upper_error.clip(lower=0)

            plt.errorbar(
                group["x_value"],
                group["y_value_50"],
                yerr=[lower_error, upper_error],
                fmt='none',
                ecolor=color,
                capsize=5,
                alpha=0.6
            )

    # 8. 装饰
    plt.title(f"Benchmark: {kernel_name} - {metric_name} ({mode})", fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(title="Implementation (Hardware)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    # 9. 保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{kernel_name}_{metric_name}_{mode}_comparison.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")
    plt.close()

def main():
    args = parse_args()
    
    # 1. 加载所有 CSV
    print(f"Loading data from: {args.csv_paths}")
    df = load_and_merge_data(args.csv_paths)
    
    # 2. 确定有哪些 Operation Modes (forward, backward, full)
    # 先根据 Kernel 和 Metric 过滤
    target_df = df[
        (df["kernel_name"] == args.kernel_name) & 
        (df["metric_name"] == args.metric_name)
    ]
    
    if target_df.empty:
        print(f"No data found for kernel '{args.kernel_name}' and metric '{args.metric_name}' in the provided CSVs.")
        return

    available_modes = target_df["kernel_operation_mode"].unique()
    print(f"Found modes: {available_modes}")

    # 3. 循环绘图
    for mode in available_modes:
        plot_benchmark(target_df, args.kernel_name, args.metric_name, mode, args.output_dir)

if __name__ == "__main__":
    main()
