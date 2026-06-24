# -*- coding: utf-8 -*-
"""
Generate FMCTO-style VEC reliability dataset from:
1) VEC_edge_data.csv
2) edge-servers-site-optus-melbCBD.csv
3) users-melbcbd-generated.csv

The script follows the construction logic described in the DOCX:
- EUA base stations are treated as edge servers.
- EUA users are treated as mobile edge devices.
- VEC records are used to sample task execution time, energy and mobility distribution.
- Candidate servers are selected by communication range.
- The actual offloading target is selected by an independent mixed behavior policy:
  random / minimum predicted delay / minimum predicted energy / minimum current load.
- Four criteria are generated: delay_score, energy_score, resource_score, link_score.
- TCRI is generated from actual delay, actual energy and task success state.

Run example:
python generate_fmcto_dataset.py \
  --vec VEC_edge_data.csv \
  --servers edge-servers-site-optus-melbCBD.csv \
  --users users-melbcbd-generated.csv \
  --out FMCTO_Dataset.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Basic utilities
# -----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def clip01(x):
    """Clip scalar or array to [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def haversine_matrix_m(
    user_lat: np.ndarray,
    user_lon: np.ndarray,
    server_lat: np.ndarray,
    server_lon: np.ndarray,
) -> np.ndarray:
    """Return distance matrix in meters, shape = [num_users, num_servers]."""
    earth_radius_m = 6_371_000.0
    lat1 = np.radians(user_lat)[:, None]
    lon1 = np.radians(user_lon)[:, None]
    lat2 = np.radians(server_lat)[None, :]
    lon2 = np.radians(server_lon)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_m * np.arcsin(np.sqrt(a))


def validate_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要字段: {missing}\n当前字段: {list(df.columns)}")


def load_inputs(vec_path: str, servers_path: str, users_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vec = pd.read_csv(vec_path)
    servers = pd.read_csv(servers_path)
    users = pd.read_csv(users_path)

    validate_columns(
        vec,
        [
            "task_id",
            "arrival_time",
            "execution_time",
            "completion_time",
            "latency",
            "energy_consumption",
            "is_offloaded",
            "vm_id",
            "device_mobility_status",
        ],
        "VEC_edge_data.csv",
    )
    validate_columns(servers, ["SITE_ID", "LATITUDE", "LONGITUDE"], "edge-servers-site-optus-melbCBD.csv")
    validate_columns(users, ["Latitude", "Longitude"], "users-melbcbd-generated.csv")

    # Keep only valid rows.
    vec = vec.dropna(subset=["execution_time", "latency", "energy_consumption", "device_mobility_status"]).copy()
    servers = servers.dropna(subset=["SITE_ID", "LATITUDE", "LONGITUDE"]).copy().reset_index(drop=True)
    users = users.dropna(subset=["Latitude", "Longitude"]).copy().reset_index(drop=True)

    return vec, servers, users


# -----------------------------
# Dataset construction
# -----------------------------

def build_dataset(
    vec: pd.DataFrame,
    servers: pd.DataFrame,
    users: pd.DataFrame,
    n_samples: int = None,
    seed: int = 42,
    comm_range_m: float = 500.0,
    deadline_factor: float = 2.0,
    gamma: float = 10.0,
    availability_threshold: float = 0.35,
    link_beta: float = 2.0,
    vmax_mps: float = 25.0,
    queue_window_ms: float = 300.0,
    wait_energy_coef: float = 2e-4,
    random_noise: bool = True,
) -> pd.DataFrame:
    """
    Construct a reliability dataset with four criteria and TCRI.

    Formula mapping:
    L_pred(k,j) = T_trans(k,j) + T_queue(j) + T_comp(k,j)
    D_k = rho * T_exec(k)
    x1 = max(0, 1 - L_pred / D_k)
    E_pred = E_trans + E_wait
    x2 = max(0, 1 - E_pred / E_budget)
    A_j = 1 - U_j
    x3 = 1 / (1 + exp(-gamma * (A_j - A0)))
    x4 = exp(-beta * (v / vmax) * (d / R_c))
    y  = min(mu_L, mu_E, mu_S)
    """
    rng = np.random.default_rng(seed)

    if n_samples is None:
        n_samples = len(vec)
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples 必须大于 0")

    num_users = len(users)
    num_servers = len(servers)

    # Distance matrix and candidate server set.
    distance_matrix = haversine_matrix_m(
        users["Latitude"].to_numpy(),
        users["Longitude"].to_numpy(),
        servers["LATITUDE"].to_numpy(),
        servers["LONGITUDE"].to_numpy(),
    )
    candidate_sets: List[np.ndarray] = []
    for u in range(num_users):
        candidates = np.where(distance_matrix[u] <= comm_range_m)[0]
        if len(candidates) == 0:
            # Safety fallback: use the nearest server if no candidate exists.
            candidates = np.array([int(np.argmin(distance_matrix[u]))])
        candidate_sets.append(candidates)

    # Server compute ability and background utilization are not provided by EUA,
    # so they are generated reproducibly. Larger compute power means faster execution.
    server_compute_power = rng.uniform(3.5, 8.5, size=num_servers)
    # Background CPU load prevents the resource score from degenerating to values near 1.
    # It represents other services running on the same edge server.
    server_background_load = rng.beta(2.0, 4.0, size=num_servers) * 0.75
    server_load_phase = rng.uniform(0.0, 2.0 * np.pi, size=num_servers)

    # Server available time for single-queue service simulation.
    server_available_time = np.zeros(num_servers, dtype=float)

    # Estimate Poisson arrivals from the original VEC time span.
    original_span = float(vec["arrival_time"].max() - vec["arrival_time"].min())
    if original_span <= 0:
        original_span = float(n_samples)
    mean_interarrival = original_span / max(n_samples, 1)
    inter_arrivals = rng.exponential(scale=mean_interarrival, size=n_samples)
    sim_arrival_time = np.cumsum(inter_arrivals)
    # Rescale to the original time range to keep units close to the VEC records.
    sim_arrival_time = sim_arrival_time / sim_arrival_time.max() * original_span
    sim_arrival_time += float(vec["arrival_time"].min())

    # Sample task records from VEC empirical distribution.
    sampled_idx = rng.integers(0, len(vec), size=n_samples)
    sampled_vec = vec.iloc[sampled_idx].reset_index(drop=True)

    # Energy budget follows the DOCX: 95% quantile of offloaded task energy.
    offloaded_energy = vec.loc[vec["is_offloaded"] == 1, "energy_consumption"]
    if len(offloaded_energy) == 0:
        offloaded_energy = vec["energy_consumption"]
    energy_budget = float(offloaded_energy.quantile(0.95))
    energy_budget = max(energy_budget, 1e-9)

    # Mobility status -> approximate speed in m/s.
    speed_map: Dict[str, float] = {
        "static": 0.5,
        "low": 5.0,
        "medium": 12.0,
        "high": 22.0,
    }

    rows = []
    policy_names = np.array(["random", "min_delay", "min_energy", "min_load"])

    for k in range(n_samples):
        task = sampled_vec.iloc[k]
        arrival = float(sim_arrival_time[k])
        user_id = int(rng.integers(0, num_users))
        candidates = candidate_sets[user_id]
        distances = distance_matrix[user_id, candidates]

        execution_time = max(float(task["execution_time"]), 1e-9)
        base_latency = max(float(task["latency"]), 1e-9)
        base_energy = max(float(task["energy_consumption"]), 1e-9)
        mobility_status = str(task["device_mobility_status"]).strip().lower()
        speed_mps = speed_map.get(mobility_status, 8.0)

        # Use latency minus execution time as empirical network overhead.
        # If negative because of dirty data, use a small positive fallback.
        empirical_network_ms = max(base_latency - execution_time, 1.0)

        # Candidate-wise predicted delay.
        distance_ratio = distances / max(comm_range_m, 1e-9)
        tx_delay_ms = empirical_network_ms * (1.0 + distance_ratio)
        queue_delay_ms = np.maximum(0.0, server_available_time[candidates] - arrival)
        comp_delay_ms = execution_time / server_compute_power[candidates]
        pred_delay_ms = tx_delay_ms + queue_delay_ms + comp_delay_ms

        # Candidate-wise predicted energy.
        # Transmission energy increases with distance; waiting energy increases with queueing time.
        tx_energy = base_energy * (0.65 + 0.35 * np.minimum(distance_ratio, 1.5) ** 2)
        wait_energy = wait_energy_coef * queue_delay_ms
        pred_energy = tx_energy + wait_energy

        # Candidate-wise CPU utilization and calibrated resource availability.
        # CPU utilization includes background load, a mild time-varying component,
        # and queue-induced pressure.
        time_load = 0.10 * (1.0 + np.sin(2.0 * np.pi * arrival / max(original_span, 1e-9) + server_load_phase[candidates])) / 2.0
        queue_load = queue_delay_ms / max(queue_window_ms, 1e-9)
        cpu_util = clip01(server_background_load[candidates] + time_load + queue_load)
        raw_availability = 1.0 - cpu_util
        resource_score_all = sigmoid(gamma * (raw_availability - availability_threshold))

        # Candidate-wise link stability.
        link_score_all = np.exp(-link_beta * (speed_mps / max(vmax_mps, 1e-9)) * distance_ratio)
        link_score_all = clip01(link_score_all)

        # Independent mixed behavior policy: 1/4 for each rule.
        policy = str(rng.choice(policy_names))
        if policy == "random":
            local_choice = int(rng.integers(0, len(candidates)))
        elif policy == "min_delay":
            local_choice = int(np.argmin(pred_delay_ms))
        elif policy == "min_energy":
            local_choice = int(np.argmin(pred_energy))
        else:  # min_load
            local_choice = int(np.argmin(cpu_util))

        server_idx = int(candidates[local_choice])
        server_id = servers.iloc[server_idx]["SITE_ID"]
        distance_m = float(distances[local_choice])

        # Scores calculated at task arrival time for the selected server.
        deadline_ms = max(deadline_factor * execution_time, 1e-9)
        selected_pred_delay = float(pred_delay_ms[local_choice])
        selected_pred_energy = float(pred_energy[local_choice])
        selected_resource_score = float(resource_score_all[local_choice])
        selected_link_score = float(link_score_all[local_choice])

        delay_score = float(max(0.0, 1.0 - selected_pred_delay / deadline_ms))
        energy_score = float(max(0.0, 1.0 - selected_pred_energy / energy_budget))
        resource_score = selected_resource_score
        link_score = selected_link_score

        # Actual feedback after execution.
        selected_queue_delay = float(queue_delay_ms[local_choice])
        selected_tx_delay = float(tx_delay_ms[local_choice])
        selected_comp_delay = float(comp_delay_ms[local_choice])
        selected_tx_energy = float(tx_energy[local_choice])
        selected_wait_energy = float(wait_energy[local_choice])

        if random_noise:
            delay_noise = float(rng.lognormal(mean=0.0, sigma=0.05))
            energy_noise = float(rng.lognormal(mean=0.0, sigma=0.03))
        else:
            delay_noise = 1.0
            energy_noise = 1.0

        actual_delay_ms = (selected_tx_delay + selected_queue_delay + selected_comp_delay) * delay_noise
        actual_energy = (selected_tx_energy + selected_wait_energy) * energy_noise

        # Link is probabilistic; overload/timeout/energy overrun may also lead to failure.
        link_survived = rng.random() <= selected_link_score
        success = int((actual_delay_ms <= deadline_ms) and (actual_energy <= energy_budget) and link_survived)

        mu_delay = float(max(0.0, 1.0 - actual_delay_ms / deadline_ms))
        mu_energy = float(max(0.0, 1.0 - actual_energy / energy_budget))
        mu_success = float(success)
        tcri = float(min(mu_delay, mu_energy, mu_success))

        # Update selected server queue state. Only computation occupies the server CPU queue.
        service_start = max(server_available_time[server_idx], arrival)
        server_available_time[server_idx] = service_start + selected_comp_delay

        rows.append(
            {
                # Basic identifiers
                "sample_id": k + 1,
                "source_task_id": int(task["task_id"]),
                "user_id": user_id,
                "server_index": server_idx,
                "server_site_id": server_id,
                "behavior_policy": policy,
                # Spatial and mobility information
                "user_latitude": float(users.iloc[user_id]["Latitude"]),
                "user_longitude": float(users.iloc[user_id]["Longitude"]),
                "server_latitude": float(servers.iloc[server_idx]["LATITUDE"]),
                "server_longitude": float(servers.iloc[server_idx]["LONGITUDE"]),
                "distance_m": distance_m,
                "mobility_status": mobility_status,
                "speed_mps": speed_mps,
                "candidate_server_count": int(len(candidates)),
                # Original/sampled task information
                "arrival_time": arrival,
                "execution_time": execution_time,
                "vec_latency": base_latency,
                "vec_energy_consumption": base_energy,
                "empirical_network_ms": empirical_network_ms,
                # Prediction components at arrival
                "deadline_ms": deadline_ms,
                "energy_budget": energy_budget,
                "transmission_delay_ms": selected_tx_delay,
                "queue_delay_ms": selected_queue_delay,
                "computation_delay_ms": selected_comp_delay,
                "pred_delay_ms": selected_pred_delay,
                "transmission_energy": selected_tx_energy,
                "waiting_energy": selected_wait_energy,
                "pred_energy": selected_pred_energy,
                "server_cpu_utilization": float(cpu_util[local_choice]),
                "raw_resource_availability": float(raw_availability[local_choice]),
                # Four criteria vector X_k
                "delay_score": delay_score,
                "energy_score": energy_score,
                "resource_score": resource_score,
                "link_score": link_score,
                # Actual feedback and label y_k
                "actual_delay_ms": actual_delay_ms,
                "actual_energy": actual_energy,
                "success": success,
                "mu_delay": mu_delay,
                "mu_energy": mu_energy,
                "mu_success": mu_success,
                "TCRI": tcri,
            }
        )

    dataset = pd.DataFrame(rows)

    # Put the most important learning fields at the front.
    front_cols = ["delay_score", "energy_score", "resource_score", "link_score", "TCRI"]
    other_cols = [c for c in dataset.columns if c not in front_cols]
    dataset = dataset[front_cols + other_cols]

    return dataset


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FMCTO-style VEC reliability dataset.")
    parser.add_argument("--vec", default="VEC_edge_data.csv", help="Path to VEC_edge_data.csv")
    parser.add_argument("--servers", default="edge-servers-site-optus-melbCBD.csv", help="Path to edge server site CSV")
    parser.add_argument("--users", default="users-melbcbd-generated.csv", help="Path to user location CSV")
    parser.add_argument("--out", default="FMCTO_Dataset.csv", help="Output CSV path")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples; default uses all VEC records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--comm_range_m", type=float, default=500.0, help="Communication range in meters")
    parser.add_argument("--deadline_factor", type=float, default=2.0, help="rho in D_k = rho * T_exec(k)")
    parser.add_argument("--gamma", type=float, default=10.0, help="Sigmoid steepness for resource score")
    parser.add_argument("--availability_threshold", type=float, default=0.35, help="A0 in resource score")
    parser.add_argument("--link_beta", type=float, default=2.0, help="Link attenuation coefficient beta")
    parser.add_argument("--vmax_mps", type=float, default=25.0, help="Maximum mobility speed")
    parser.add_argument("--queue_window_ms", type=float, default=300.0, help="Window for converting queue delay to CPU utilization")
    parser.add_argument("--wait_energy_coef", type=float, default=2e-4, help="Waiting energy coefficient")
    parser.add_argument("--no_noise", action="store_true", help="Disable lognormal noise for actual feedback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vec_path = Path(args.vec)
    servers_path = Path(args.servers)
    users_path = Path(args.users)
    out_path = Path(args.out)

    vec, servers, users = load_inputs(str(vec_path), str(servers_path), str(users_path))

    dataset = build_dataset(
        vec=vec,
        servers=servers,
        users=users,
        n_samples=args.n_samples,
        seed=args.seed,
        comm_range_m=args.comm_range_m,
        deadline_factor=args.deadline_factor,
        gamma=args.gamma,
        availability_threshold=args.availability_threshold,
        link_beta=args.link_beta,
        vmax_mps=args.vmax_mps,
        queue_window_ms=args.queue_window_ms,
        wait_energy_coef=args.wait_energy_coef,
        random_noise=not args.no_noise,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("数据集构建完成")
    print(f"输出文件: {out_path.resolve()}")
    print(f"样本数: {len(dataset)}")
    print("核心字段: delay_score, energy_score, resource_score, link_score, TCRI")
    print("\n核心字段统计:")
    print(dataset[["delay_score", "energy_score", "resource_score", "link_score", "TCRI"]].describe().round(4))
    print("\n前5行:")
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
