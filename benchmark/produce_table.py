import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass

import pandas as pd

# 定义数据路径
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/all_benchmark_data.csv"))
# 输出路径改为 reports 而不是 visualizations
REPORTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports/"))


@dataclass
class ReportConfig:
    """
    Configuration for the report generation script.
    """
    kernel_name: str
    metric_name: str
    kernel_operation_mode: str = "full"
    extra_config_filter: str | None = None
    save_csv: bool = False
    print_markdown: bool = True


def parse_args() -> ReportConfig:
    parser = ArgumentParser()
    parser.add_argument("--kernel-name", type=str, required=True, help="Kernel name to benchmark")
    parser.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="Metric name to analyze (speed/memory)",
    )
    parser.add_argument(
        "--kernel-operation-mode",
        type=str,
        nargs="*",
        default=None,
        help="Kernel operation modes (forward/backward/full).",
    )
    parser.add_argument(
        "--extra-config-filter",
        type=str,
        default=None,
        help="Filter for extra_benchmark_config (e.g., \"'H': 4096\").",
    )
    parser.add_argument("--save-csv", action="store_true", help="Save the resulting table to a CSV file")
    parser.add_argument("--no-markdown", action="store_false", dest="print_markdown", help="Do not print markdown table to console")

    args = parser.parse_args()
    return args


def load_data(config: ReportConfig) -> pd.DataFrame:
    """
    Loads and filters data. (Logic kept mostly same as original to ensure consistency)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)

    base_filtered_df = df[
        (df["kernel_name"] == config.kernel_name)
        & (df["metric_name"] == config.metric_name)
        & (df["kernel_operation_mode"] == config.kernel_operation_mode)
    ]

    if base_filtered_df.empty:
        raise ValueError(
            f"No data found for {config.kernel_name} | {config.metric_name} | {config.kernel_operation_mode}"
        )

    unique_configs = base_filtered_df["extra_benchmark_config_str"].unique()
    selected_config_str = None

    # Handle Config Filtering
    if config.extra_config_filter:
        matched = [c for c in unique_configs if config.extra_config_filter in c]
        if matched:
            selected_config_str = matched[0]
            if len(matched) > 1:
                print(f"Warning: Multiple configs matched '{config.extra_config_filter}'. Using: {selected_config_str}")
        else:
            # Fallback logic
            if len(unique_configs) > 0:
                selected_config_str = unique_configs[0]
                print(f"Warning: Filter '{config.extra_config_filter}' not found. Defaulting to: {selected_config_str}")
    elif len(unique_configs) > 0:
        selected_config_str = unique_configs[0]
        if len(unique_configs) > 1:
            print(f"Warning: Multiple configs found. Defaulting to: {selected_config_str}")

    if selected_config_str:
        final_df = base_filtered_df[base_filtered_df["extra_benchmark_config_str"] == selected_config_str].copy()
    else:
        final_df = base_filtered_df.copy()

    # Clean up numeric columns
    for col in ["y_value_20", "y_value_50", "y_value_80"]:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
            
    return final_df


def generate_comparison_table(df: pd.DataFrame, config: ReportConfig):
    """
    Generates a comparison table between different kernel providers.
    """
    # 1. Prepare Data
    # We use y_value_50 (median) as the primary metric for comparison
    x_label = df["x_label"].iloc[0]
    unit = df["metric_unit"].iloc[0]
    
    # Pivot: Index=InputSize, Columns=Provider, Values=Median
    pivot_df = df.pivot_table(
        index="x_value", 
        columns="kernel_provider", 
        values="y_value_50",
        aggfunc="mean" # Should be unique, but mean adds safety
    )
    
    pivot_df.index.name = x_label
    
    # Sort index strictly numeric if possible
    try:
        pivot_df.index = pivot_df.index.astype(float)
        pivot_df = pivot_df.sort_index()
    except ValueError:
        pass # Keep as string if not numeric

    # 2. Calculate Comparisons (Speedup or Memory Saving)
    providers = pivot_df.columns.tolist()
    
    # Identify Liger Kernel name (assuming it contains "liger" case-insensitive)
    liger_cols = [c for c in providers if "liger" in c.lower()]
    liger_col = liger_cols[0] if liger_cols else None
    
    comparison_stats = pivot_df.copy()
    
    # Rename columns to include units
    comparison_stats.columns = [f"{c} ({unit})" for c in comparison_stats.columns]

    if liger_col and len(providers) > 1:
        others = [p for p in providers if p != liger_col]
        baseline_col = others[0] # Pick the first non-liger as baseline (e.g., HF or Triton)
        
        liger_vals = pivot_df[liger_col]
        baseline_vals = pivot_df[baseline_col]

        if config.metric_name == "speed":
            # Higher is worse (time), so Speedup = Baseline / Liger
            # Example: HF=10ms, Liger=5ms -> Speedup = 2.0x
            speedup_col = f"Speedup ({baseline_col} / {liger_col})"
            comparison_stats[speedup_col] = (baseline_vals / liger_vals).apply(lambda x: f"{x:.2f}x")
            print(f"Computed Speedup comparing '{liger_col}' against baseline '{baseline_col}'")
            
        elif config.metric_name == "memory":
            # Lower is better. Savings = (Baseline - Liger) / Baseline
            # Example: HF=100MB, Liger=80MB -> Savings = 20%
            saving_col = f"Mem Savings ({liger_col} vs {baseline_col})"
            comparison_stats[saving_col] = ((baseline_vals - liger_vals) / baseline_vals * 100).apply(lambda x: f"{x:.2f}%")
            print(f"Computed Memory Savings comparing '{liger_col}' against baseline '{baseline_col}'")

    # 3. Output
    print("\n" + "="*60)
    print(f"BENCHMARK REPORT: {config.kernel_name} - {config.kernel_operation_mode}")
    print(f"Metric: {config.metric_name}")
    print("="*60 + "\n")

    if config.print_markdown:
        print(comparison_stats.to_markdown())
    else:
        print(comparison_stats)

    if config.save_csv:
        os.makedirs(REPORTS_PATH, exist_ok=True)
        filename = f"{config.kernel_name}_{config.metric_name}_{config.kernel_operation_mode}.csv"
        out_path = os.path.join(REPORTS_PATH, filename)
        comparison_stats.to_csv(out_path)
        print(f"\nReport saved to: {out_path}")


def main():
    args = parse_args()
    
    # Determine operation modes to run
    if args.metric_name == "memory":
        # Memory usually doesn't strictly have fwd/bwd separation in standard logs, 
        # or implies "full" usage. Adjust based on your CSV structure.
        modes = ["full"] 
    elif args.kernel_operation_mode:
        modes = args.kernel_operation_mode
    else:
        # Auto-detect available modes
        all_df = pd.read_csv(DATA_PATH)
        filtered = all_df[(all_df["kernel_name"] == args.kernel_name) & (all_df["metric_name"] == args.metric_name)]
        modes = filtered["kernel_operation_mode"].unique().tolist()
        if not modes:
            print(f"No data found for kernel '{args.kernel_name}' and metric '{args.metric_name}'.", file=sys.stderr)
            sys.exit(1)

    for mode in modes:
        config = ReportConfig(
            kernel_name=args.kernel_name,
            metric_name=args.metric_name,
            kernel_operation_mode=mode,
            extra_config_filter=args.extra_config_filter,
            save_csv=args.save_csv,
            print_markdown=args.print_markdown
        )
        try:
            df = load_data(config)
            generate_comparison_table(df, config)
        except ValueError as e:
            print(f"Skipping {mode}: {e}")

if __name__ == "__main__":
    main()
