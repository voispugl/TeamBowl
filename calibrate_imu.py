#!/usr/bin/env python3
"""
IMU Allan variance calibration script.

Usage:
    python3 calibrate_imu.py <path_to_bag>

Records /imu/data (Xsens Sirius 300) and /oak/imu/data (OAK-D Pro W),
computes Allan variance, and prints ready-to-paste YAML for:
  - state_estimation.yaml  process_noise_covariance
  - xsens_mti_node.yaml    angular_velocity_stddev / linear_acceleration_stddev
"""

import sys
import math
from pathlib import Path
from datetime import datetime

import numpy as np


# ── rosbag2 reader ───────────────────────────────────────────────────────────

def read_bag(bag_path: str, topics: list[str]) -> dict[str, list]:
    """Read IMU messages from a rosbag2 bag. Returns {topic: [(t_sec, msg), ...]}"""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Imu
    except ImportError:
        sys.exit("ERROR: rosbag2_py or rclpy not available. Source your ROS workspace first:\n"
                 "  source /opt/ros/humble/setup.bash\n"
                 "  source ~/TeamBowl/teambowl_ws/install/setup.bash")

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    data = {t: [] for t in topics}

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic not in topics:
            continue
        msg = deserialize_message(raw, Imu)
        t_sec = t_ns * 1e-9
        data[topic].append((t_sec, msg))

    for topic in topics:
        if not data[topic]:
            print(f"WARNING: No messages found on {topic}")

    return data


# ── Allan variance ────────────────────────────────────────────────────────────

def allan_deviation(data: np.ndarray, dt: float, n_tau: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute overlapping Allan deviation.

    Args:
        data: 1-D array of measurements (rad/s or m/s²)
        dt:   sample interval in seconds
        n_tau: number of averaging time points (log-spaced)

    Returns:
        taus:  averaging times (seconds)
        adevs: Allan deviation at each tau
    """
    n = len(data)
    max_m = n // 2
    ms = np.unique(np.logspace(0, np.log10(max_m), n_tau).astype(int))
    ms = ms[ms >= 1]

    taus = []
    adevs = []
    for m in ms:
        tau = m * dt
        # Overlapping Allan variance
        chunks = n - 2 * m
        if chunks < 1:
            break
        sums = np.cumsum(data)
        sums = np.concatenate([[0], sums])
        avgs = (sums[2*m:] - 2*sums[m:n-m+1] + sums[:n-2*m+1]) / m
        avar = np.mean(avgs**2) / 2.0
        taus.append(tau)
        adevs.append(math.sqrt(avar))

    return np.array(taus), np.array(adevs)


def extract_noise_params(taus: np.ndarray, adevs: np.ndarray) -> dict:
    """
    Extract key noise parameters from Allan deviation curve.

    Returns dict with:
        arw:              Angle/Velocity Random Walk (at tau=1s), units/√s
        bias_instability: minimum of the curve, units
        rrw:              Rate Random Walk (slope at long tau), units·√s
    """
    if len(taus) == 0:
        return {'arw': float('nan'), 'bias_instability': float('nan'), 'rrw': float('nan')}

    # ARW: interpolate Allan deviation at tau=1s (or nearest available)
    if taus[0] <= 1.0 <= taus[-1]:
        arw = float(np.interp(1.0, taus, adevs))
    else:
        arw = float(adevs[np.argmin(np.abs(taus - 1.0))])

    # Bias instability: minimum of the curve
    bias_instability = float(np.min(adevs))

    # Rate random walk: slope in log-log at long tau (tau > tau_at_minimum)
    min_idx = np.argmin(adevs)
    if min_idx < len(taus) - 2:
        long_taus = taus[min_idx:]
        long_adevs = adevs[min_idx:]
        # fit log-log slope; RRW has slope +0.5
        slope, _ = np.polyfit(np.log10(long_taus), np.log10(long_adevs), 1)
        tau_rrw = 3.0  # evaluate at tau=3s for RRW coefficient
        rrw = float(np.interp(tau_rrw, taus, adevs)) / math.sqrt(tau_rrw) if tau_rrw <= taus[-1] else float('nan')
    else:
        rrw = float('nan')

    return {'arw': arw, 'bias_instability': bias_instability, 'rrw': rrw}


# ── Per-IMU analysis ──────────────────────────────────────────────────────────

def analyse_imu(name: str, messages: list) -> dict:
    """Analyse one IMU's messages. Returns noise parameter dict."""
    if not messages:
        print(f"\n  [{name}] No data — skipping.")
        return None

    times = np.array([t for t, _ in messages])
    dt = float(np.median(np.diff(times)))
    rate_hz = 1.0 / dt

    gx = np.array([m.angular_velocity.x for _, m in messages])
    gy = np.array([m.angular_velocity.y for _, m in messages])
    gz = np.array([m.angular_velocity.z for _, m in messages])
    ax = np.array([m.linear_acceleration.x for _, m in messages])
    ay = np.array([m.linear_acceleration.y for _, m in messages])
    az = np.array([m.linear_acceleration.z for _, m in messages])

    duration_min = (times[-1] - times[0]) / 60.0
    n = len(times)

    print(f"\n  [{name}]")
    print(f"    Messages : {n}  |  Rate: {rate_hz:.1f} Hz  |  Duration: {duration_min:.1f} min")
    print(f"    Gyro bias (rad/s): x={np.mean(gx):.5f}  y={np.mean(gy):.5f}  z={np.mean(gz):.5f}")
    print(f"    Accel bias (m/s²): x={np.mean(ax):.4f}  y={np.mean(ay):.4f}  z={np.mean(az):.4f}")

    if duration_min < 5.0:
        print(f"    WARNING: Only {duration_min:.1f} min of data. Allan variance needs 30+ min for accuracy.")

    results = {}
    for axis, data, label in [
        ('gx', gx, 'gyro_x'), ('gy', gy, 'gyro_y'), ('gz', gz, 'gyro_z'),
        ('ax', ax, 'accel_x'), ('ay', ay, 'accel_y'), ('az', az, 'accel_z'),
    ]:
        taus, adevs = allan_deviation(data, dt)
        params = extract_noise_params(taus, adevs)
        results[axis] = params

    # Summarise gyro
    arw_gyro = np.mean([results['gx']['arw'], results['gy']['arw'], results['gz']['arw']])
    bi_gyro  = np.mean([results['gx']['bias_instability'], results['gy']['bias_instability'], results['gz']['bias_instability']])
    arw_accel = np.mean([results['ax']['arw'], results['ay']['arw'], results['az']['arw']])
    bi_accel  = np.mean([results['ax']['bias_instability'], results['ay']['bias_instability'], results['az']['bias_instability']])

    arw_gyro_deg_sqrthr = math.degrees(arw_gyro) * 60.0  # rad/√s → °/√hr
    bi_gyro_deg_hr = math.degrees(bi_gyro) * 3600.0       # rad/s → °/hr

    print(f"    Gyro ARW       : {arw_gyro:.2e} rad/√s  ({arw_gyro_deg_sqrthr:.3f} °/√hr)")
    print(f"    Gyro bias inst : {bi_gyro:.2e} rad/s    ({bi_gyro_deg_hr:.3f} °/hr)")
    print(f"    Accel ARW (VRW): {arw_accel:.2e} m/s²/√s")
    print(f"    Accel bias inst: {bi_accel:.2e} m/s²")

    return {
        'dt': dt,
        'rate_hz': rate_hz,
        'arw_gyro': arw_gyro,
        'bi_gyro': bi_gyro,
        'arw_accel': arw_accel,
        'bi_accel': bi_accel,
        'per_axis': results,
        'gyro_bias': (float(np.mean(gx)), float(np.mean(gy)), float(np.mean(gz))),
        'accel_bias': (float(np.mean(ax)), float(np.mean(ay)), float(np.mean(az))),
    }


# ── EKF Q matrix computation ──────────────────────────────────────────────────

def compute_ekf_q(xsens: dict, oak: dict, ekf_hz: float = 50.0) -> list[float]:
    """
    Compute the 15-element diagonal of the EKF process noise covariance Q.

    State order: x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw, ax, ay, az

    Strategy:
      - Orientation (roll/pitch/yaw): use bias_instability² / ekf_hz (how fast orientation drifts)
      - Angular velocity (vroll/vpitch/vyaw): use ARW_gyro² × ekf_hz (noise per step)
      - Linear velocity (vx/vy/vz): use ARW_accel² × ekf_hz integrated to velocity
      - Acceleration (ax/ay/az): use ARW_accel² × ekf_hz
      - Position (x/y/z): loose, driven by velocity integration
    """
    def safe(val, fallback):
        return val if val is not None and not math.isnan(val) else fallback

    arw_g  = safe(xsens['arw_gyro'],  5e-5) if xsens else 5e-5
    bi_g   = safe(xsens['bi_gyro'],   1e-5) if xsens else 1e-5
    arw_a  = safe(xsens['arw_accel'], 8e-4) if xsens else 8e-4
    bi_a   = safe(xsens['bi_accel'],  1e-4) if xsens else 1e-4

    q_pos   = 0.05                         # x, y, z — loose, from wheel integration
    q_ori   = bi_g**2 / ekf_hz            # roll, pitch, yaw
    q_vel   = (arw_a**2) * ekf_hz         # vx, vy, vz
    q_gyro  = (arw_g**2) * ekf_hz         # vroll, vpitch, vyaw
    q_accel = (arw_a**2) * ekf_hz         # ax, ay, az

    # Clamp to reasonable bounds
    q_ori   = max(1e-8, min(q_ori,  1e-2))
    q_gyro  = max(1e-8, min(q_gyro, 1e-2))
    q_vel   = max(1e-6, min(q_vel,  0.1))
    q_accel = max(1e-6, min(q_accel, 0.1))

    diag = [
        q_pos,  q_pos,  0.06,   # x, y, z
        q_ori,  q_ori,  q_ori,  # roll, pitch, yaw
        q_vel,  q_vel,  0.06,   # vx, vy, vz
        q_gyro, q_gyro, q_gyro, # vroll, vpitch, vyaw
        q_accel, q_accel, q_accel,  # ax, ay, az
    ]
    return diag


def fmt_q(diag: list[float]) -> str:
    """Format a 15-element diagonal as a YAML 15x15 flat matrix."""
    labels = ['x', 'y', 'z', 'ro', 'pi', 'ya', 'vx', 'vy', 'vz', 'vro', 'vpi', 'vya', 'ax', 'ay', 'az']
    rows = []
    for i, (val, lbl) in enumerate(zip(diag, labels)):
        row = ['0.0'] * 15
        row[i] = f'{val:.3e}'
        prefix = '      ' if i > 0 else '      '
        rows.append(f'{prefix}[{", ".join(row)}]  # {lbl}')
    return '    process_noise_covariance: [\n' + ',\n'.join(rows) + ']'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bag_path = sys.argv[1]
    if not Path(bag_path).exists():
        sys.exit(f"ERROR: Bag path not found: {bag_path}")

    print(f"\n{'='*60}")
    print(f"  IMU Allan Variance Calibration")
    print(f"  Bag: {bag_path}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    print("\n[1/3] Reading bag...")
    data = read_bag(bag_path, ['/imu/data', '/oak/imu/data'])

    print("\n[2/3] Computing Allan variance...")
    xsens = analyse_imu('Xsens Sirius 300 (/imu/data)', data['/imu/data'])
    oak   = analyse_imu('OAK-D Pro W (/oak/imu/data)', data['/oak/imu/data'])

    print(f"\n[3/3] Computing EKF parameters...")
    ekf_hz = 50.0
    q_diag = compute_ekf_q(xsens, oak, ekf_hz)

    # Xsens stddev values (from per-axis ARW)
    if xsens:
        gx_arw = xsens['per_axis']['gx']['arw']
        gy_arw = xsens['per_axis']['gy']['arw']
        gz_arw = xsens['per_axis']['gz']['arw']
        ax_arw = xsens['per_axis']['ax']['arw']
        ay_arw = xsens['per_axis']['ay']['arw']
        az_arw = xsens['per_axis']['az']['arw']
        gx_bi  = xsens['per_axis']['gx']['bias_instability']
        gy_bi  = xsens['per_axis']['gy']['bias_instability']
        gz_bi  = xsens['per_axis']['gz']['bias_instability']
    else:
        gx_arw = gy_arw = gz_arw = 5e-5
        ax_arw = ay_arw = az_arw = 8e-4
        gx_bi  = gy_bi  = gz_bi  = 1e-5

    now = datetime.now().strftime('%Y-%m-%d')
    xsens_arw_deg = math.degrees(xsens['arw_gyro']) * 60 if xsens else float('nan')
    xsens_bi_deg  = math.degrees(xsens['bi_gyro']) * 3600 if xsens else float('nan')
    oak_arw_deg   = math.degrees(oak['arw_gyro']) * 60 if oak else float('nan')
    oak_bi_deg    = math.degrees(oak['bi_gyro']) * 3600 if oak else float('nan')

    print(f"\n{'='*60}")
    print("  RESULTS — paste into state_estimation.yaml")
    print(f"{'='*60}\n")
    print(f"    # IMU calibration — {now} — bag: {Path(bag_path).name}")
    print(f"    # Xsens Sirius 300: ARW={xsens_arw_deg:.3f} °/√hr, bias_instability={xsens_bi_deg:.3f} °/hr")
    print(f"    # OAK-D Pro W:      ARW={oak_arw_deg:.3f} °/√hr, bias_instability={oak_bi_deg:.3f} °/hr")
    print(fmt_q(q_diag))

    print(f"\n{'='*60}")
    print("  RESULTS — paste into xsens_mti_node.yaml")
    print(f"{'='*60}\n")
    print(f"        # Calibrated {now} from Allan variance on {Path(bag_path).name}")
    print(f"        angular_velocity_stddev: [{gx_arw:.4e}, {gy_arw:.4e}, {gz_arw:.4e}]  # rad/s (ARW)")
    print(f"        linear_acceleration_stddev: [{ax_arw:.4e}, {ay_arw:.4e}, {az_arw:.4e}]  # m/s² (VRW)")
    print(f"        orientation_stddev: [{gx_bi:.4e}, {gy_bi:.4e}, {gz_bi:.4e}]  # rad (bias instability)")

    print(f"\n{'='*60}")
    print("  Next steps:")
    print("  1. Paste process_noise_covariance into:")
    print("       teambowl_ws/src/state_estimation/config/state_estimation.yaml")
    print("  2. Paste stddev values into:")
    print("       teambowl_ws/src/drivers/xsens_mti_ros2_driver/param/xsens_mti_node.yaml")
    print("  3. Restart the stack and verify /odometry/filtered is stable when stationary.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
