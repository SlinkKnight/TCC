"""
Kalman 1D Interativo — Pitch (gyro + acelerômetro)
===================================================
Agora com visualização do acelerômetro explícita.

Uso:
    python kalman_pitch_interativo.py --csv dados_imu.csv
"""

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ═══════════════════════════════════════
# Kalman 1D
# ═══════════════════════════════════════
class Kalman1D:
    def __init__(
        self,
        Q_angle=0.001,
        Q_bias=0.003,
        R_measure=0.03,
        angle0=0.0,
        bias0=0.0,
        P00=0.0,
        P01=0.0,
        P10=0.0,
        P11=0.0
    ):
        self.Q_angle = Q_angle
        self.Q_bias = Q_bias
        self.R_measure = R_measure

        self.angle = angle0
        self.bias = bias0
        self.rate = 0.0

        self.P = np.array([
            [P00, P01],
            [P10, P11]
        ], dtype=float)

    def update(self, newAngle, newRate, dt):

        # predict
        self.rate = newRate - self.bias
        self.angle += dt * self.rate

        self.P[0][0] += dt * (
            dt * self.P[1][1]
            - self.P[0][1]
            - self.P[1][0]
            + self.Q_angle
        )
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_bias * dt

        # update
        y = newAngle - self.angle
        S = self.P[0][0] + self.R_measure

        K0 = self.P[0][0] / S
        K1 = self.P[1][0] / S

        self.angle += K0 * y
        self.bias += K1 * y

        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]

        self.P[0][0] -= K0 * P00_temp
        self.P[0][1] -= K0 * P01_temp
        self.P[1][0] -= K1 * P00_temp
        self.P[1][1] -= K1 * P01_temp

        return self.angle


# ═══════════════════════════════════════
# Execução do filtro
# ═══════════════════════════════════════
def rodar_kalman(df, **p):

    kf = Kalman1D(**p)

    pitch_raw = []
    pitch_kf = []
    pitch_acc = []

    raw = p["angle0"]

    for _, row in df.iterrows():

        ax = row["ax_ms2"]
        ay = row["ay_ms2"]
        az = row["az_ms2"]

        gyroY = math.degrees(row["gy_rads"])
        dt = row["dt_s"]

        # acelerômetro
        acc_pitch = math.degrees(
            math.atan2(-ax, math.sqrt(ay*ay + az*az))
        )

        pitch_acc.append(acc_pitch)

        # gyro integrado
        raw += gyroY * dt
        pitch_raw.append(raw)

        # kalman
        kf_val = kf.update(acc_pitch, gyroY, dt)
        pitch_kf.append(kf_val)

    return np.array(pitch_raw), np.array(pitch_kf), np.array(pitch_acc)


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main(csv_path):

    df = pd.read_csv(csv_path)
    t = np.cumsum(df["dt_s"].values)

    defaults = {
        "Q_angle": 0.001,
        "Q_bias": 0.003,
        "R_measure": 0.03,
        "angle0": 0.0,
        "bias0": 0.0,
        "P00": 0.0,
        "P01": 0.0,
        "P10": 0.0,
        "P11": 0.0
    }

    raw, kf, acc = rodar_kalman(df, **defaults)

    # ═══════════════════════
    # FIGURA
    # ═══════════════════════
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=0.08, bottom=0.55)

    ax.set_title("Kalman 1D — Pitch (gyro + acelerômetro)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo (graus)")
    ax.grid(True)

    line_raw, = ax.plot(t, raw, '--', lw=1, label="Gyro integrado")
    line_acc, = ax.plot(t, acc, lw=1, alpha=0.7, label="Acelerômetro")
    line_kf,  = ax.plot(t, kf, lw=2, label="Kalman")

    ax.legend()

    # ═══════════════════════
    # SLIDERS
    # ═══════════════════════
    sliders = {}

    cfg = [
        ("Q_angle", 0.000001, 0.05),
        ("Q_bias", 0.000001, 0.05),
        ("R_measure", 0.0001, 1.0),

        ("angle0", -180, 180),
        ("bias0", -50, 50),

        ("P00", 0.0, 10.0),
        ("P01", -10.0, 10.0),
        ("P10", -10.0, 10.0),
        ("P11", 0.0, 10.0),
    ]

    y0 = 0.48
    step = 0.045

    for i, (name, vmin, vmax) in enumerate(cfg):
        ax_s = plt.axes([0.15, y0 - i*step, 0.7, 0.025])
        sliders[name] = Slider(ax_s, name, vmin, vmax, valinit=defaults[name])

    # ═══════════════════════
    # UPDATE
    # ═══════════════════════
    def update(_):

        params = {k: s.val for k, s in sliders.items()}

        raw_new, kf_new, acc_new = rodar_kalman(df, **params)

        line_raw.set_ydata(raw_new)
        line_kf.set_ydata(kf_new)
        line_acc.set_ydata(acc_new)

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    plt.show()


# ═══════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    main(args.csv)