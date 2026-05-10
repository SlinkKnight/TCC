"""
Visualização do filtro Kalman do ESP32
======================================

Mostra:
- Roll do acelerômetro
- Roll filtrado (Kalman)
- Roll suavizado

- Pitch do acelerômetro
- Pitch filtrado (Kalman)

- Yaw integrado do giroscópio

Uso:
    python visualizar_kalman.py --csv dados.csv

Salvar:
    python visualizar_kalman.py --csv dados.csv --save grafico.png
"""

import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════════════════════════════════════
# Parâmetros do filtro (iguais ao ESP32)
# ═════════════════════════════════════════════════════════════════════════════
Q_ANGLE = 0.001
Q_BIAS = 0.003
R_MEASURE = 0.03

LP_ALPHA_X = 0.25

PONTO_ZERO_Y = -12


# ═════════════════════════════════════════════════════════════════════════════
# Kalman 1D
# ═════════════════════════════════════════════════════════════════════════════
def kalman_update(new_angle,
                  new_rate,
                  dt,
                  angle,
                  bias,
                  P):

    # predição
    angle += dt * (new_rate - bias)

    # covariância
    P[0][0] += dt * (
        dt * P[1][1]
        - P[0][1]
        - P[1][0]
        + Q_ANGLE
    )

    P[0][1] -= dt * P[1][1]
    P[1][0] -= dt * P[1][1]
    P[1][1] += Q_BIAS * dt

    # ganho de Kalman
    S = P[0][0] + R_MEASURE

    K0 = P[0][0] / S
    K1 = P[1][0] / S

    # inovação
    y = new_angle - angle

    if y > 180:
        y -= 360

    if y < -180:
        y += 360

    # update
    angle += K0 * y
    bias += K1 * y

    P00 = P[0][0]
    P01 = P[0][1]

    P[0][0] -= K0 * P00
    P[0][1] -= K0 * P01
    P[1][0] -= K1 * P00
    P[1][1] -= K1 * P01

    return angle, bias, P


# ═════════════════════════════════════════════════════════════════════════════
# Processamento
# ═════════════════════════════════════════════════════════════════════════════
def process(csv_path, save_path=None):

    df = pd.read_csv(csv_path)

    required = {
        "ax_ms2",
        "ay_ms2",
        "az_ms2",
        "gx_rads",
        "gy_rads",
        "gz_rads",
        "dt_s"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    # rad/s → deg/s
    gx = np.degrees(df["gx_rads"].values)
    gy = np.degrees(df["gy_rads"].values)
    gz = np.degrees(df["gz_rads"].values)

    ax = df["ax_ms2"].values
    ay = df["ay_ms2"].values
    az = df["az_ms2"].values

    dt = df["dt_s"].values

    n = len(df)

    # ═════════════════════════════════════════
    # Estados
    # ═════════════════════════════════════════
    kalmanAngleX = 0.0
    kalmanAngleY = 0.0

    biasX = 0.0
    biasY = 0.0

    P_X = np.zeros((2, 2))
    P_Y = np.zeros((2, 2))

    yaw = 0.0

    smoothRoll = 0.0

    roll_offset = 0.0
    pitch_offset = 0.0

    referenceSet = False

    # ═════════════════════════════════════════
    # Buffers
    # ═════════════════════════════════════════
    accRoll_buf = []
    accPitch_buf = []

    roll_buf = []
    pitch_buf = []
    yaw_buf = []

    smooth_buf = []

    # ═════════════════════════════════════════
    # Loop
    # ═════════════════════════════════════════
    for i in range(n):

        # acelerômetro
        accAngleX = math.degrees(
            math.atan2(ay[i], az[i])
        )

        accAngleY = math.degrees(
            math.atan2(
                -ax[i],
                math.sqrt(ay[i]*ay[i] + az[i]*az[i])
            )
        )

        # referência inicial
        if not referenceSet:

            roll_offset = accAngleX
            pitch_offset = accAngleY

            smoothRoll = accAngleX

            referenceSet = True

        # remove offset
        accAngleX -= roll_offset

        accAngleY = (
            accAngleY
            - pitch_offset
            + PONTO_ZERO_Y
        )

        # ═════════════════════════════════════
        # Roll Kalman
        # ═════════════════════════════════════
        roll, biasX, P_X = kalman_update(
            accAngleX,
            gx[i],
            dt[i],
            kalmanAngleX,
            biasX,
            P_X
        )

        kalmanAngleX = roll

        # ═════════════════════════════════════
        # Pitch Kalman
        # ═════════════════════════════════════
        pitch, biasY, P_Y = kalman_update(
            accAngleY,
            gy[i],
            dt[i],
            kalmanAngleY,
            biasY,
            P_Y
        )

        kalmanAngleY = pitch

        # ═════════════════════════════════════
        # Yaw integrado
        # ═════════════════════════════════════
        yaw += gz[i] * dt[i]

        # ═════════════════════════════════════
        # smoothing do roll
        # ═════════════════════════════════════
        smoothRoll = (
            LP_ALPHA_X * roll
            + (1.0 - LP_ALPHA_X) * smoothRoll
        )

        # ═════════════════════════════════════
        # salva
        # ═════════════════════════════════════
        accRoll_buf.append(accAngleX)
        accPitch_buf.append(accAngleY)

        roll_buf.append(roll)
        pitch_buf.append(pitch)
        yaw_buf.append(yaw)

        smooth_buf.append(smoothRoll)

    # ═════════════════════════════════════════
    # Tempo
    # ═════════════════════════════════════════
    t = np.cumsum(dt)

    # ═════════════════════════════════════════
    # Plot
    # ═════════════════════════════════════════
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True
    )

    fig.suptitle(
        "Filtro Kalman ESP32",
        fontsize=16,
        fontweight="bold"
    )

    # ─────────────────────────────────────────
    # ROLL
    # ─────────────────────────────────────────
    axs[0].plot(
        t,
        accRoll_buf,
        label="Accel Roll",
        alpha=0.5
    )

    axs[0].plot(
        t,
        roll_buf,
        label="Kalman Roll",
        linewidth=2
    )

    axs[0].plot(
        t,
        smooth_buf,
        label="Smooth Roll",
        linewidth=2
    )

    axs[0].set_title("ROLL")
    axs[0].set_ylabel("graus")
    axs[0].legend()
    axs[0].grid(True)

    # ─────────────────────────────────────────
    # PITCH
    # ─────────────────────────────────────────
    axs[1].plot(
        t,
        accPitch_buf,
        label="Accel Pitch",
        alpha=0.5
    )

    axs[1].plot(
        t,
        pitch_buf,
        label="Kalman Pitch",
        linewidth=2
    )

    axs[1].set_title("PITCH")
    axs[1].set_ylabel("graus")
    axs[1].legend()
    axs[1].grid(True)

    # ─────────────────────────────────────────
    # YAW
    # ─────────────────────────────────────────
    axs[2].plot(
        t,
        yaw_buf,
        label="Integrated Gyro Z",
        linewidth=2
    )

    axs[2].set_title("YAW")
    axs[2].set_ylabel("graus")
    axs[2].set_xlabel("tempo (s)")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

    # ═════════════════════════════════════════
    # Estatísticas
    # ═════════════════════════════════════════
    print("\n══════════════════════════════")

    print(f"Amostras: {n}")
    print(f"Duração: {t[-1]:.2f} s")

    print("\nROLL")
    print(f"  mínimo: {min(roll_buf):.2f}")
    print(f"  máximo: {max(roll_buf):.2f}")

    print("\nPITCH")
    print(f"  mínimo: {min(pitch_buf):.2f}")
    print(f"  máximo: {max(pitch_buf):.2f}")

    print("\nYAW")
    print(f"  drift final: {yaw_buf[-1]:.2f}°")

    print("══════════════════════════════\n")

    # ═════════════════════════════════════════
    # salvar
    # ═════════════════════════════════════════
    if save_path:

        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        print(f"Figura salva: {save_path}")

    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV da IMU"
    )

    parser.add_argument(
        "--save",
        default=None,
        help="Salvar imagem"
    )

    args = parser.parse_args()

    process(
        args.csv,
        args.save
    )