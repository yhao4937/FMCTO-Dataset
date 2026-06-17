# -*- coding: utf-8 -*-
"""
根据 DOCX 中“面向模糊测度学习的可靠协同卸载数据集构造”方法，
将 VEC_edge_data.csv 构造为包含四个准则评分 x1-x4 与综合可靠性标签 y 的新数据集。

直接运行：
    python build_vec_reliability_dataset.py

指定输入输出：
    python build_vec_reliability_dataset.py --input VEC_edge_data.csv --output VEC_reliability_dataset.csv
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ========== 1. 参数设置：对应文档中的公式参数 ==========
RHO = 1.5                 # 时延容忍系数 rho，D(k)=rho*T_exec(k)
E_BUDGET = 2.0            # 能耗预算 E_budget
LAMBDA_RESOURCE = 0.5     # 资源可用性均衡因子 lambda
BETA = 0.1                # 链路稳定性指数衰减系数 beta
TAU = 0.1                 # TCRI 中 Sigmoid 温度系数 tau
WINDOW_W = 1000.0         # 资源可用性滑动时间窗口 W；文档未给固定数值，这里设为可调整默认值
COLD_START_SCORE = 0.5    # 某服务器无历史记录时的冷启动中性评分

# 移动状态到速度的映射：static=0, low=2, medium=7, high=15
MOBILITY_TO_SPEED = {
    "static": 0.0,
    "low": 2.0,
    "medium": 7.0,
    "high": 15.0,
}

REQUIRED_COLUMNS = [
    "task_id",
    "arrival_time",
    "execution_time",
    "completion_time",
    "latency",
    "energy_consumption",
    "is_offloaded",
    "vm_id",
    "device_mobility_status",
]


def stable_sigmoid_negative(z: np.ndarray) -> np.ndarray:
    """
    计算 1 / (1 + exp(z))，用于 y = 1/(1+exp((L-D)/(tau*D)))。
    对 z 做截断，避免 exp 溢出。
    """
    z_clip = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(z_clip))


def check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "输入CSV缺少必要字段：{}\n当前字段为：{}".format(missing, list(df.columns))
        )


def build_dataset(
    input_path: str,
    output_path: str,
    rho: float = RHO,
    e_budget: float = E_BUDGET,
    lambda_resource: float = LAMBDA_RESOURCE,
    beta: float = BETA,
    tau: float = TAU,
    window_w: float = WINDOW_W,
    cold_start_score: float = COLD_START_SCORE,
) -> pd.DataFrame:
    """读取原始 VEC 数据，构造可靠协同卸载监督学习数据集。"""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path.resolve()}")

    df = pd.read_csv(input_path)
    check_columns(df)

    # ========== 2. 基础预处理 ==========
    # 仅保留卸载任务，并按到达时间升序排列，保证资源可用性计算不使用未来信息。
    data = df[df["is_offloaded"] == 1].copy()
    data = data.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)

    # 数值字段转换，避免字符串类型导致计算错误。
    numeric_cols = [
        "arrival_time",
        "execution_time",
        "completion_time",
        "latency",
        "energy_consumption",
        "vm_id",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # 移动状态映射为速度值 v(k)。
    mobility = data["device_mobility_status"].astype(str).str.lower().str.strip()
    data["velocity_mps"] = mobility.map(MOBILITY_TO_SPEED)

    # 删除无法计算的异常样本。
    before_drop = len(data)
    data = data.dropna(subset=numeric_cols + ["velocity_mps"]).reset_index(drop=True)
    dropped = before_drop - len(data)

    # ========== 3. 计算 D(k)、x1、x2、x4 与 y ==========
    # 软截止时间 D(k)=rho*T_exec(k)
    data["deadline"] = rho * data["execution_time"]

    # 时延评分 x1(k)=max(0, 1-L(k)/D(k))，并截断到[0,1]
    data["x1_delay_score"] = (1.0 - data["latency"] / data["deadline"]).clip(lower=0.0, upper=1.0)

    # 能耗评分 x2(k)=max(0, 1-E(k)/E_budget)，并截断到[0,1]
    data["x2_energy_score"] = (1.0 - data["energy_consumption"] / e_budget).clip(lower=0.0, upper=1.0)

    # 链路稳定性评分 x4(k)=exp(-beta*v(k))
    data["x4_link_stability_score"] = np.exp(-beta * data["velocity_mps"]).clip(0.0, 1.0)

    # 综合可靠性标签 y(k)=1/(1+exp((L(k)-D(k))/(tau*D(k))))
    z = (data["latency"] - data["deadline"]) / (tau * data["deadline"])
    data["y_tcri"] = stable_sigmoid_negative(z.to_numpy())

    # ========== 4. 动态计算资源可用性 x3 ==========
    # 对每条记录，只使用同一服务器在当前任务到达前、且已完成、且处于窗口W内的历史任务。
    history_by_vm = defaultdict(list)
    completion_rate_values = []
    efficiency_ratio_values = []
    resource_score_values = []
    history_count_values = []

    for idx, row in data.iterrows():
        vm_id = row["vm_id"]
        arrival_time = row["arrival_time"]

        candidates = []
        for hist_idx in history_by_vm[vm_id]:
            hist = data.loc[hist_idx]
            # 已完成：历史任务完成时间早于当前任务到达时间；
            # 窗口内：完成时间不早于 arrival_time-W。
            if (hist["completion_time"] < arrival_time) and (hist["completion_time"] >= arrival_time - window_w):
                candidates.append(hist_idx)

        if len(candidates) == 0:
            # 冷启动：没有可用历史时，使用中性分数，避免引入未来信息。
            cr = cold_start_score
            ei = cold_start_score
            hist_count = 0
        else:
            hist_df = data.loc[candidates]
            # CR(k)：历史任务中满足 L<=D 的比例。
            cr = float((hist_df["latency"] <= hist_df["deadline"]).mean())

            # EI(k)：当前执行时长与近期平均执行时长之比，并截断到1。
            # 对应文档公式 EI(k)=min(1, T_exec(k)/mean(T_exec_H))。
            mean_exec_h = float(hist_df["execution_time"].mean())
            if mean_exec_h <= 0:
                ei = cold_start_score
            else:
                ei = min(1.0, float(row["execution_time"] / mean_exec_h))
            hist_count = len(candidates)

        x3 = lambda_resource * cr + (1.0 - lambda_resource) * ei

        completion_rate_values.append(cr)
        efficiency_ratio_values.append(ei)
        resource_score_values.append(x3)
        history_count_values.append(hist_count)

        # 当前任务进入该服务器历史池。后续任务是否使用它，由 completion_time 和 window_w 判断。
        history_by_vm[vm_id].append(idx)

    data["resource_history_count"] = history_count_values
    data["completion_rate_CR"] = completion_rate_values
    data["execution_efficiency_EI"] = efficiency_ratio_values
    data["x3_resource_availability_score"] = resource_score_values

    # ========== 5. 整理输出字段 ==========
    # supervised learning 核心字段：X=(x1,x2,x3,x4), y=y_tcri。
    output_cols = [
        "task_id",
        "arrival_time",
        "completion_time",
        "execution_time",
        "latency",
        "energy_consumption",
        "vm_id",
        "device_mobility_status",
        "velocity_mps",
        "deadline",
        "resource_history_count",
        "completion_rate_CR",
        "execution_efficiency_EI",
        "x1_delay_score",
        "x2_energy_score",
        "x3_resource_availability_score",
        "x4_link_stability_score",
        "y_tcri",
    ]

    new_dataset = data[output_cols].copy()

    # 为便于后续机器学习建模，也保留简洁别名列。
    new_dataset["x1"] = new_dataset["x1_delay_score"]
    new_dataset["x2"] = new_dataset["x2_energy_score"]
    new_dataset["x3"] = new_dataset["x3_resource_availability_score"]
    new_dataset["x4"] = new_dataset["x4_link_stability_score"]
    new_dataset["y"] = new_dataset["y_tcri"]

    # 保留6位小数，避免CSV过长，同时不影响建模。
    float_cols = new_dataset.select_dtypes(include=["float64", "float32"]).columns
    new_dataset[float_cols] = new_dataset[float_cols].round(6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_dataset.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("新数据集构造完成")
    print(f"原始样本数：{len(df)}")
    print(f"卸载样本数：{int((df['is_offloaded'] == 1).sum())}")
    print(f"有效输出样本数：{len(new_dataset)}")
    print(f"因缺失/异常被删除样本数：{dropped}")
    print(f"输出文件：{output_path.resolve()}")
    print("\n核心字段统计：")
    print(new_dataset[["x1", "x2", "x3", "x4", "y"]].describe().round(6))

    return new_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="构造可靠协同卸载监督学习数据集")
    parser.add_argument("--input", default="VEC_edge_data.csv", help="原始CSV路径")
    parser.add_argument("--output", default="VEC_reliability_dataset.csv", help="输出CSV路径")
    parser.add_argument("--rho", type=float, default=RHO, help="时延容忍系数rho")
    parser.add_argument("--e-budget", type=float, default=E_BUDGET, help="能耗预算E_budget")
    parser.add_argument("--lambda-resource", type=float, default=LAMBDA_RESOURCE, help="资源评分均衡因子lambda")
    parser.add_argument("--beta", type=float, default=BETA, help="链路稳定性衰减系数beta")
    parser.add_argument("--tau", type=float, default=TAU, help="TCRI温度系数tau")
    parser.add_argument("--window-w", type=float, default=WINDOW_W, help="资源可用性滑动时间窗口W")
    parser.add_argument("--cold-start-score", type=float, default=COLD_START_SCORE, help="无历史记录时的冷启动评分")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        input_path=args.input,
        output_path=args.output,
        rho=args.rho,
        e_budget=args.e_budget,
        lambda_resource=args.lambda_resource,
        beta=args.beta,
        tau=args.tau,
        window_w=args.window_w,
        cold_start_score=args.cold_start_score,
    )
