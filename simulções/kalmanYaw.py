"""
Kalman Filter Linear — Pitch e Yaw com correção magnética
==========================================================
Lê o CSV gerado por gravar_serial.py, aplica um Filtro de Kalman
(4 estados: pitch, yaw, bias_pitch, bias_yaw) e plota a comparação:
  • linha tracejada = integração bruta do giroscópio
  • linha sólida    = estimativa do Kalman (pitch corrigido por acelerômetro,
                      yaw corrigido por magnetômetro com compensação de inclinação)

Uso:
    python kalman_pitch_yaw.py --csv dados_imu.csv
    python kalman_pitch_yaw.py --csv dados_imu.csv --salvar resultado.png
"""

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# Parâmetros do Filtro de Kalman (linear)
# ══════════════════════════════════════════════════════════════════════════════
Q_ANGLE_PSD = 1e-4          # (rad²/s)  densidade espectral do ruído de ângulo
Q_BIAS_PSD  = 1e-5          # ((rad/s)²/s) densidade espectral do bias
R_PITCH     = 0.01          # (rad²)   variância da medição do pitch (acelerômetro)
R_YAW       = 0.03          # (rad²)   variância da medição do yaw (magnetômetro)

G_NORM     = 9.80665
ACC_THRESH = 0.5            # m/s² — tolerância para usar acelerômetro como referência
MAG_THRESH = 0.1            # µT? apenas para evitar divisão por zero (não usado diretamente)


# ══════════════════════════════════════════════════════════════════════════════
# Filtro de Kalman linear (estados: pitch, yaw, bias_pitch, bias_yaw)
# ══════════════════════════════════════════════════════════════════════════════
class LinearKalman:
    def __init__(self):
        # Estado: [pitch (rad), yaw (rad), bias_pitch (rad/s), bias_yaw (rad/s)]
        self.x = np.zeros(4)
        # Covariância inicial
        self.P = np.diag([0.01, 0.01, 0.1, 0.1])

    def predict(self, gyro_pitch_rate, gyro_yaw_rate, dt):
        """
        gyro_pitch_rate : ω_y (rad/s)
        gyro_yaw_rate    : ω_z (rad/s)
        dt               : intervalo de tempo (s)
        """
        # Matriz de transição
        F = np.array([
            [1, 0, -dt, 0],
            [0, 1, 0, -dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Matriz de entrada
        B = np.array([
            [dt, 0],
            [0, dt],
            [0, 0],
            [0, 0]
        ])
        u = np.array([gyro_pitch_rate, gyro_yaw_rate])

        self.x = F @ self.x + B @ u

        # Ruído de processo discretizado
        Q = np.diag([
            Q_ANGLE_PSD * dt,
            Q_ANGLE_PSD * dt,
            Q_BIAS_PSD  * dt,
            Q_BIAS_PSD  * dt
        ])
        self.P = F @ self.P @ F.T + Q

    def update_pitch(self, pitch_meas):
        """Atualização com medição do pitch (acelerômetro)"""
        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        z = np.array([pitch_meas])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R_PITCH
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_yaw(self, yaw_meas):
        """Atualização com medição do yaw (magnetômetro compensado)"""
        H = np.array([[0.0, 1.0, 0.0, 0.0]])
        z = np.array([yaw_meas])
        y = z - H @ self.x
        # Ajuste para saltos de ±π na diferença de yaw
        y[0] = math.atan2(math.sin(y[0]), math.cos(y[0]))
        S = H @ self.P @ H.T + R_YAW
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_pitch_deg(self):
        return math.degrees(self.x[0])

    def get_yaw_deg(self):
        return math.degrees(self.x[1])


# ══════════════════════════════════════════════════════════════════════════════
# Integração bruta do giroscópio (sem filtro)
# ══════════════════════════════════════════════════════════════════════════════
def integrar_bruto(df):
    pitch_raw = [0.0]
    yaw_raw   = [0.0]
    p, y = 0.0, 0.0
    for _, row in df.iterrows():
        dt = row["dt_s"]
        p += math.degrees(row["gy_rads"]) * dt
        y += math.degrees(row["gz_rads"]) * dt
        pitch_raw.append(p)
        yaw_raw.append(y)
    return np.array(pitch_raw[:-1]), np.array(yaw_raw[:-1])


# ══════════════════════════════════════════════════════════════════════════════
# Cálculo do yaw a partir do magnetômetro com compensação de inclinação
# ══════════════════════════════════════════════════════════════════════════════
def yaw_from_magnetometer(mx, my, mz, pitch_rad, roll_rad):
    """
    Aplica compensação de inclinação usando pitch e roll estimados.
    Retorna yaw em radianos (faixa -pi a pi).
    """
    # Corrigir campos para o plano horizontal
    mx_c = mx * math.cos(pitch_rad) + my * math.sin(pitch_rad) * math.sin(roll_rad) \
           + mz * math.sin(pitch_rad) * math.cos(roll_rad)
    my_c = my * math.cos(roll_rad) - mz * math.sin(roll_rad)

    yaw = math.atan2(-my_c, mx_c)   # convenção: yaw = atan2(-my, mx)
    return yaw


def roll_from_accelerometer(ax, ay, az, g_norm=G_NORM):
    """Estima roll (rad) a partir do acelerômetro, assumindo repouso."""
    # Roll = atan2(ay, az)
    return math.atan2(ay, az)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════
def processar(csv_path: str, salvar: str | None):
    # Leitura do CSV
    df = pd.read_csv(csv_path)
    required = {"ax_ms2","ay_ms2","az_ms2",
                "gx_rads","gy_rads","gz_rads",
                "mx_ut","my_ut","mz_ut","dt_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    n = len(df)
    print(f"[Kalman] {n} amostras carregadas de '{csv_path}'")

    # Sinal bruto
    pitch_raw, yaw_raw = integrar_bruto(df)

    # Filtro de Kalman
    kf = LinearKalman()
    pitch_kf = np.zeros(n)
    yaw_kf   = np.zeros(n)

    # Para evitar saltos no yaw medido, manter referência contínua (opcional)
    last_yaw_meas = None
    yaw_offset = 0.0

    for i, row in df.iterrows():
        ax, ay, az = row["ax_ms2"], row["ay_ms2"], row["az_ms2"]
        gx, gy, gz = row["gx_rads"], row["gy_rads"], row["gz_rads"]
        mx, my, mz = row["mx_ut"], row["my_ut"], row["mz_ut"]
        dt = row["dt_s"]

        # Predição usando giroscópio
        kf.predict(gy, gz, dt)

        # Medição do pitch (acelerômetro) se a norma estiver próxima de g
        acc_norm = math.sqrt(ax*ax + ay*ay + az*az)
        if abs(acc_norm - G_NORM) <= ACC_THRESH:
            pitch_meas = math.atan2(-ax, math.sqrt(ay*ay + az*az))
            kf.update_pitch(pitch_meas)

            # Também podemos atualizar o roll (apenas para compensação magnética)
            roll_rad = roll_from_accelerometer(ax, ay, az)
        else:
            # Se não tivermos aceleração confiável, usar o último roll conhecido
            # Para simplificar, reutilizamos o roll calculado na última iteração válida
            # (inicialmente zero)
            if i == 0:
                roll_rad = 0.0
            else:
                roll_rad = roll_from_accelerometer(ax, ay, az)  # ainda pode ser usado

        # Medição do yaw via magnetômetro (sempre que houver campos magnéticos razoáveis)
        # Obs: o magnetômetro pode ser ruidoso; aplicamos correção mesmo assim.
        # É possível adicionar uma condição de intensidade mínima, mas não obrigatório.
        yaw_meas = yaw_from_magnetometer(mx, my, mz, kf.x[0], roll_rad)

        # Ajuste de continuidade: evitar saltos de 2π na medição (opcional)
        if last_yaw_meas is not None:
            diff = yaw_meas - last_yaw_meas
            if diff > math.pi:
                yaw_offset -= 2*math.pi
            elif diff < -math.pi:
                yaw_offset += 2*math.pi
        yaw_meas_corrected = yaw_meas + yaw_offset
        last_yaw_meas = yaw_meas

        kf.update_yaw(yaw_meas_corrected)

        pitch_kf[i] = kf.get_pitch_deg()
        yaw_kf[i]   = kf.get_yaw_deg()

        if i % 500 == 0:
            print(f"  processando... {i}/{n}", end="\r")

    print(f"[Kalman] Concluído.{' '*20}")

    # Eixo de tempo
    t = np.cumsum(df["dt_s"].values)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT — apenas sobreposição (pitch e yaw)
    # ══════════════════════════════════════════════════════════════════════════
    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "grid.linestyle":   "--",
    })

    fig, (ax_p, ax_y) = plt.subplots(2, 1, figsize=(12, 8),
                                      facecolor="#0f1117")
    fig.suptitle("Filtro de Kalman Linear — Pitch e Yaw com correção magnética",
                 color="white", fontsize=14, fontweight="bold", y=0.96)

    COR_RAW = "#FF6B6B"   # vermelho
    COR_KF  = "#4ECDC4"   # teal
    BG_AX   = "#1a1d27"

    def estilo_ax(ax, titulo):
        ax.set_facecolor(BG_AX)
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.set_title(titulo, color="#dddddd", fontsize=10, pad=8)
        ax.set_xlabel("tempo (s)", color="#888888", fontsize=9)
        ax.set_ylabel("graus (°)", color="#888888", fontsize=9)
        ax.legend(fontsize=9, facecolor="#22253a", labelcolor="white",
                  edgecolor="#444")

    # Pitch
    ax_p.plot(t, pitch_raw, color=COR_RAW, lw=0.8, alpha=0.7,
              linestyle="--", label="Bruto (gyro integrado)")
    ax_p.plot(t, pitch_kf, color=COR_KF, lw=1.4, label="Kalman (acelerômetro)")
    estilo_ax(ax_p, "PITCH — comparação")

    # Yaw
    ax_y.plot(t, yaw_raw, color=COR_RAW, lw=0.8, alpha=0.7,
              linestyle="--", label="Bruto (gyro integrado)")
    ax_y.plot(t, yaw_kf, color=COR_KF, lw=1.4,
              label="Kalman (magnetômetro + compensação)")
    estilo_ax(ax_y, "YAW — comparação")

    # Legenda única (opcional)
    fig.legend(
        handles=[
            Line2D([0], [0], color=COR_RAW, lw=1.4, linestyle="--",
                   label="Integração pura do giroscópio (drift)"),
            Line2D([0], [0], color=COR_KF, lw=1.4,
                   label="Filtro de Kalman (fusão com acelerômetro/magnetômetro)")
        ],
        loc="lower center", ncol=2,
        fontsize=9, facecolor="#22253a", labelcolor="white",
        edgecolor="#444", bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Estatísticas rápidas
    print(f"\n{'─'*50}")
    print(f"  Pitch — drift bruto:      {pitch_raw[-1]:.2f}°")
    print(f"  Pitch — Kalman final:     {pitch_kf[-1]:.2f}°")
    print(f"  Yaw   — drift bruto:      {yaw_raw[-1]:.2f}°")
    print(f"  Yaw   — Kalman final:     {yaw_kf[-1]:.2f}°")
    print(f"{'─'*50}\n")

    if salvar:
        fig.savefig(salvar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Kalman] Figura salva em '{salvar}'")

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtro de Kalman Linear com magnetômetro para yaw")
    parser.add_argument("--csv", required=True,
                        help="Arquivo CSV gerado pelo gravar_serial.py")
    parser.add_argument("--salvar", default=None,
                        help="Salva o gráfico em arquivo (ex: resultado.png)")
    args = parser.parse_args()

    processar(args.csv, args.salvar)