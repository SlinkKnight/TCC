"""
Gravador Serial → CSV
=====================
Lê dados do ESP32 (CSV) e salva em arquivo.

Agora com:
- acelerômetro
- giroscópio
- magnetômetro

Formato esperado:
ax,ay,az,gx,gy,gz,mx,my,mz,dt

Uso:
    python gravar_serial.py --port COM3
    python gravar_serial.py --port /dev/ttyUSB0 --segundos 30
"""

import argparse
import csv
import signal
import sys
import time

import serial

# ── Colunas esperadas ──────────────────────────────────────────────────────
COLUNAS = [
    "ax_ms2",
    "ay_ms2",
    "az_ms2",

    "gx_rads",
    "gy_rads",
    "gz_rads",

    "mx_ut",
    "my_ut",
    "mz_ut",

    "dt_s"
]

# ── Estado global ──────────────────────────────────────────────────────────
_rodando = True


def _parar(sig, frame):
    global _rodando
    _rodando = False


def parse_linha(linha: str):
    """
    Recebe uma linha CSV da serial.

    Retorna:
        lista de 10 floats
    ou:
        None se inválida
    """

    linha = linha.strip()

    if not linha:
        return None

    if linha.startswith("#"):
        return None

    try:
        vals = [float(v) for v in linha.split(",")]

        # Agora esperamos 10 valores:
        # ax ay az gx gy gz mx my mz dt
        if len(vals) != 10:
            return None

        return vals

    except ValueError:
        return None


def gravar(port: str, baud: int, arquivo: str, segundos: float | None):

    global _rodando

    try:
        ser = serial.Serial(port, baud, timeout=1)

    except serial.SerialException as e:
        print(f"[ERRO] Não foi possível abrir {port}: {e}")
        sys.exit(1)

    # ESP32 normalmente reinicia ao abrir serial
    time.sleep(2)

    ser.reset_input_buffer()

    signal.signal(signal.SIGINT, _parar)

    t_inicio = time.time()

    n_amostras = 0
    n_erros = 0

    print(f"[REC] Gravando em '{arquivo}' — Ctrl+C para encerrar", end="")

    if segundos:
        print(f" (limite: {segundos}s)")
    else:
        print()

    with open(arquivo, "w", newline="") as f:

        writer = csv.writer(f)

        # Cabeçalho CSV
        writer.writerow(COLUNAS)

        while _rodando:

            # ── Timeout ────────────────────────────────────────────────────
            if segundos and (time.time() - t_inicio) >= segundos:
                print(f"\n[REC] Tempo limite atingido ({segundos}s).")
                break

            # ── Leitura serial ────────────────────────────────────────────
            try:
                raw = ser.readline().decode(
                    "utf-8",
                    errors="ignore"
                )

            except serial.SerialException as e:
                print(f"\n[ERRO] Serial: {e}")
                break

            vals = parse_linha(raw)

            if vals is None:

                if raw.strip() and not raw.strip().startswith("#"):
                    n_erros += 1

                continue

            # ── Salva CSV ─────────────────────────────────────────────────
            writer.writerow([f"{v:.6f}" for v in vals])

            n_amostras += 1

            # ── Feedback ──────────────────────────────────────────────────
            if n_amostras % 100 == 0:

                elapsed = time.time() - t_inicio

                taxa = (
                    n_amostras / elapsed
                    if elapsed > 0 else 0
                )

                print(
                    f"\r[REC] "
                    f"{n_amostras:6d} amostras  |  "
                    f"{taxa:.1f} Hz  |  "
                    f"{elapsed:.1f}s  |  "
                    f"erros: {n_erros}   ",
                    end="",
                    flush=True
                )

    ser.close()

    elapsed = time.time() - t_inicio

    print(
        f"\n[REC] Finalizado: "
        f"{n_amostras} amostras em "
        f"{elapsed:.1f}s  →  '{arquivo}'"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gravador serial IMU → CSV"
    )

    parser.add_argument(
        "--port",
        required=True,
        help="Porta serial (ex: COM3)"
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate"
    )

    parser.add_argument(
        "--out",
        default="dados_imu.csv",
        help="Arquivo CSV de saída"
    )

    parser.add_argument(
        "--segundos",
        type=float,
        default=None,
        help="Duração máxima da gravação"
    )

    args = parser.parse_args()

    gravar(
        args.port,
        args.baud,
        args.out,
        args.segundos
    )