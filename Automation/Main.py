"""
A3OF master closed-loop controller.

This script coordinates Bayesian optimization, OT-2 liquid handling,
pre-detection computer vision, droplet characterization, and ESP32-driven
motion hardware.
"""

import os
import sys
import time
import math
import traceback
import numpy as np
import pandas as pd
import serial

# ============================================================================
# 0. Global configuration
# ============================================================================
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BO-test.csv")
MAX_ITERATIONS = 200
PRE_DETECT_SIGNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pre_detect_signal.txt")
CHARA_SIGNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chara_signal.txt")

# ============================================================================
# 1. Optimizer import and fallback
# ============================================================================
try:
    from Optimizer import build_bo_domain, bo_recommend, random_recommend
    BOFIRE_AVAILABLE = True
    print("Optimizer.py imported successfully.")
except ImportError as e:
    BOFIRE_AVAILABLE = False

    # Fallback when Optimizer.py or BoFire cannot be imported.
    def random_recommend():
        import random
        surfactants_water = ["ODEA", "LDEA", "CDEA"]
        ratios = [0, 0.0002, 0.001, 0.005]
        oils = ["PMX-10","PMX-20","PMX-50","PMX-100","PMX-200",
                "PMX-500","PMX-1000","PMX-30000","PMX-60000",
                "7500","FC-40","PFPE","mineral","PMX-350"]
        surfactants_oil = ["AEO-5","Pico-surf","Perfluoro","TEGO-410"]
        ions = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        row = {
            "SW": random.choice(surfactants_water),
            "Rwater": random.choice(ratios),
            "ROil": random.choice(ratios),
            "Oil": random.choice(oils),
            "SO": random.choice(surfactants_oil),
            "Ion": random.choice(ions),
        }
        return pd.DataFrame([row])

    def bo_recommend(csv_path):
        return random_recommend()


# ============================================================================
# 2. Plate map and valid recipe values
# ============================================================================
pure_oils = {
    "PMX-10": "A1", "PMX-20": "B1", "PMX-50": "C1", "PMX-100": "D1",
    "PMX-200": "E1", "PMX-500": "F1", "PMX-1000": "G1", "PMX-30000": "H1",
    "PMX-60000": "A2", "7500": "B2", "FC-40": "C2", "PFPE": "D2",
    "mineral": "E2", "PMX-350": "F2"
}
pure_so = {
    "AEO-5": "A3", "Pico-surf": "B3", "Perfluoro": "C3", "TEGO-410": "D3"
}
pure_sw = {
    "ODEA": "A4", "LDEA": "B4", "CDEA": "C4"
}
ion_waters = {
    0.0: "A5", 0.5: "B5", 1.0: "C5", 1.5: "D5",
    2.0: "E5", 2.5: "F5", 3.0: "G5"
}
valid_ratios = [0.0, 0.0002, 0.001, 0.005]

# A6 through H12 provide 56 dilution wells.
dilution_wells = [f"{row}{col}" for row in "ABCDEFGH" for col in range(6, 13)]

# ============================================================================
# 3. Serial port configuration
# ============================================================================
PORT_MOTOR = 'COM3'   # linear-stage and gripper ESP32
PORT_MNP   = 'COM6'   # MNP transport ESP32
PORT_OT2   = 'COM7'   # PC ESP32 bridge via ESP-NOW to OT-2
BAUD_RATE  = 115200

ser_motor = None
ser_mnp = None
ser_ot2 = None


def connect_hardware():
    global ser_motor, ser_mnp, ser_ot2
    try:
        ser_motor = serial.Serial(PORT_MOTOR, BAUD_RATE, timeout=1)
        ser_mnp   = serial.Serial(PORT_MNP, BAUD_RATE, timeout=1)
        ser_ot2   = serial.Serial(PORT_OT2, BAUD_RATE, timeout=1)
        time.sleep(2)
    except Exception as e:
        raise


def disconnect_hardware():
    for ser, name in [(ser_motor, "COM3-linear-stage"), (ser_mnp, "COM6-MNP"), (ser_ot2, "COM7-OT2")]:
        if ser and ser.is_open:
            try:
                ser.close()
            except:
                pass


# ============================================================================
# 4. Communication helpers
# ============================================================================
def send_cmd(ser, command, wait_time):
    ser.reset_input_buffer()
    ser.write((command + '\n').encode('utf-8'))
    print(f"  [{time.strftime('%H:%M:%S')}] PC -> {ser.port}: {command}")
    if wait_time > 0:
        time.sleep(wait_time)
    ser.reset_input_buffer()


def send_cmd_wait_done(ser, command, timeout=300):
    ser.reset_input_buffer()
    ser.write((command + '\n').encode('utf-8'))
    print(f"  [{time.strftime('%H:%M:%S')}] PC -> {ser.port}: {command} (waiting for DONE...)")

    start_time = time.time()
    while True:
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8', errors='ignore').strip()
            if "DONE_OT2" in response:
                elapsed = time.time() - start_time
                print(f"  [{time.strftime('%H:%M:%S')}] DONE received after {elapsed:.1f}s.")
                break
            elif response != "":
                print(f"    [OT-2 response]: {response}")
        if time.time() - start_time > timeout:
            print(f"  [{time.strftime('%H:%M:%S')}] ERROR: timeout while waiting for DONE ({timeout}s).")
            break
        time.sleep(0.05)


def ask_cv_for_result(signal_file=PRE_DETECT_SIGNAL):
    """Request pre-detection CV and block until RESULT_TRUE or RESULT_FALSE."""
    print(f"  [{time.strftime('%H:%M:%S')}] Requesting pre-detection CV check...")

    with open(signal_file, 'w') as f:
        f.write("CHECK_NOW")

    while True:
        try:
            with open(signal_file, 'r') as f:
                status = f.read().strip()
            if status == "RESULT_TRUE":
                with open(signal_file, 'w') as f:
                    f.write("WAITING")
                return True
            elif status == "RESULT_FALSE":
                with open(signal_file, 'w') as f:
                    f.write("WAITING")
                return False
        except Exception:
            pass
        time.sleep(0.2)


def ask_chara_for_reward(signal_file=CHARA_SIGNAL):
    """Request AutoChara CV scoring and block until a reward result is available."""
    print(f"  [{time.strftime('%H:%M:%S')}] Requesting droplet characterization score...")

    with open(signal_file, 'w') as f:
        f.write("EVALUATE")

    while True:
        try:
            with open(signal_file, 'r') as f:
                status = f.read().strip()
            if status.startswith("RESULT_"):
                if status == "RESULT_FAIL":
                    print(f"  [{time.strftime('%H:%M:%S')}] AutoChara found no valid droplet; reward=0.")
                    with open(signal_file, 'w') as f:
                        f.write("WAITING")
                    return 0.0
                else:
                    reward_str = status.replace("RESULT_", "")
                    reward = float(reward_str)
                    print(f"  [{time.strftime('%H:%M:%S')}] Reward = {reward:.4f}")
                    with open(signal_file, 'w') as f:
                        f.write("WAITING")
                    return reward
        except Exception:
            pass
        time.sleep(0.2)


def emit_stop():
    """Emergency-stop all motor controllers."""
    if ser_motor:
        send_cmd(ser_motor, "s", 0.5)
    if ser_mnp:
        send_cmd(ser_mnp, "stop", 0.5)
    print("Emergency stop command sent.")


# ============================================================================
# 5. Single experiment workflow
# ============================================================================

def execute_single_experiment(params, cycle, csv_file, row_index, dilution_index):

    print(f"\n{'='*60}")
    print(f"Starting experiment cycle {cycle}")
    print(f"Recipe: {params}")
    print(f"{'='*60}")

    target_oil_well = pure_oils.get(params["OT"])
    target_so_well  = pure_so.get(params["SO"])
    target_sw_well  = pure_sw.get(params["SW"])
    target_ion_well = ion_waters.get(params["Ion"])
    sco_value = params["SCO"]
    scw_value = params["SCW"]

    errors = []
    if not target_oil_well: errors.append(f"Unknown oil phase: {params['OT']}")
    if not target_so_well:  errors.append(f"Unknown oil-phase surfactant: {params['SO']}")
    if not target_sw_well:  errors.append(f"Unknown water-phase surfactant: {params['SW']}")
    if not target_ion_well: errors.append(f"Unknown ion concentration: {params['Ion']}")
    if sco_value not in valid_ratios: errors.append(f"Invalid SCO value: {sco_value}")
    if scw_value not in valid_ratios: errors.append(f"Invalid SCW value: {scw_value}")

    if errors:
        print("\nParameter error:")
        for e in errors:
            print(f"   - {e}")
        print("   Skipping this cycle with reward=0.")
        df_update = pd.read_csv(csv_file)
        df_update.loc[row_index, 'reward'] = 0
        df_update.to_csv(csv_file, index=False)
        return 0.0, dilution_index

    # ========================================================================
    # Stage 1: pre-detection
    # ========================================================================
    print(f"\n--- Stage 1: Pre-detection ---")
    cmd = f"{target_oil_well}_shiguan_oil_&_no_home"
    send_cmd_wait_done(ser_ot2, cmd, timeout=300)

    cmd = f"{target_so_well}_shiguan_surfactant_&_no_home"
    send_cmd_wait_done(ser_ot2, cmd, timeout=300)

    is_separated = ask_cv_for_result()

    if is_separated:
        reward = 0
        print("  Pre-detection found clear phase separation; reward=0 and this cycle stops early.")

        df_update = pd.read_csv(csv_file)
        df_update.loc[row_index, 'reward'] = reward
        df_update.to_csv(csv_file, index=False)
        print(f"  reward={reward} written to {csv_file}, row {row_index}.")

        if cycle % 5 == 0:
            print(f"\nCycle {cycle} ended. Please manually replace or refill consumables.")
            input("   Press Enter after consumables are ready...")
            return 0.0, dilution_index
        else:
            send_cmd(ser_motor, "m 3 1 10000", 1)
            send_cmd(ser_motor, "m 2 1 9600", 2)
            send_cmd(ser_motor, "g 35", 3)
            send_cmd(ser_motor, "m 1 0 42400", 22)
            send_cmd(ser_motor, "m 3 0 10000", 9)
            send_cmd(ser_motor, "g 53", 1.5)
            send_cmd(ser_motor, "m 3 1 22000", 10)
            send_cmd(ser_motor, "m 1 1 36700", 18)

            if cycle % 5 == 1:
                send_cmd(ser_motor, "m 2 1 2000", 3.5)
                send_cmd(ser_motor, "m 3 0 12000", 10)
                send_cmd(ser_motor, "g 35", 2)
                send_cmd(ser_motor, "m 2 1 13500", 7.5)
                send_cmd(ser_motor, "m 3 0 8000", 5.5)
                send_cmd(ser_motor, "g 53", 2)
                send_cmd(ser_motor, "m 3 1 20000", 11)
                send_cmd(ser_motor, "m 2 0 15900", 8)
                send_cmd(ser_motor, "m 1 0 37000", 7)
                send_cmd(ser_motor, "m 3 0 16000", 12)
                send_cmd(ser_motor, "g 35", 2)
                send_cmd(ser_motor, "m 1 1 42700", 20)
                send_cmd(ser_motor, "m 2 0 9200", 6)
                send_cmd(ser_motor, "m 3 0 6000", 4)
            elif cycle % 5 == 2:
                send_cmd(ser_motor, "m 2 1 15900", 8)
                send_cmd(ser_motor, "m 3 0 12000", 10)
                send_cmd(ser_motor, "g 35", 2)
                send_cmd(ser_motor, "m 2 1 10000", 6)
                send_cmd(ser_motor, "m 3 0 8000", 6.5)
                send_cmd(ser_motor, "g 53", 1)
                send_cmd(ser_motor, "m 3 1 20000", 13)
                send_cmd(ser_motor, "m 2 0 26300", 10)
                send_cmd(ser_motor, "m 1 0 37000", 13.5)
                send_cmd(ser_motor, "m 3 0 16000", 12.5)
                send_cmd(ser_motor, "g 35", 2)
                send_cmd(ser_motor, "m 1 1 42700", 20)
                send_cmd(ser_motor, "m 2 0 9200", 6)
                send_cmd(ser_motor, "m 3 0 6000", 4)
            elif cycle % 5 == 3:
                send_cmd(ser_motor, "m 2 1 25800", 12)
                send_cmd(ser_motor, "m 3 0 12000", 10)
                send_cmd(ser_motor, "g 35", 2)
                send_cmd(ser_motor, "m 2 1 9800", 6.5)
                send_cmd(ser_motor, "m 3 0 8000", 6.5)
                send_cmd(ser_motor, "g 53", 1)
                send_cmd(ser_motor, "m 3 1 20000", 13)
                send_cmd(ser_motor, "m 2 0 36000", 15)
                send_cmd(ser_motor, "m 1 0 37000", 14)
                send_cmd(ser_motor, "m 3 0 16000", 12)
                send_cmd(ser_motor, "g 35", 1.5)
                send_cmd(ser_motor, "m 1 1 42700", 20)
                send_cmd(ser_motor, "m 2 0 9200", 6)
                send_cmd(ser_motor, "m 3 0 6000", 4)
            elif cycle % 5 == 4:
                send_cmd(ser_motor, "m 2 1 35700", 15)
                send_cmd(ser_motor, "m 3 0 12000", 10)
                send_cmd(ser_motor, "g 35", 2)
                send_cmd(ser_motor, "m 2 1 9700", 6)
                send_cmd(ser_motor, "m 3 0 8000", 6.5)
                send_cmd(ser_motor, "g 53", 0.5)
                send_cmd(ser_motor, "m 3 1 20000", 13)
                send_cmd(ser_motor, "m 2 0 45800", 20)
                send_cmd(ser_motor, "m 1 0 37000", 14)
                send_cmd(ser_motor, "m 3 0 16000", 12.5)
                send_cmd(ser_motor, "g 35", 1.5)
                send_cmd(ser_motor, "m 1 1 42700", 20)
                send_cmd(ser_motor, "m 2 0 9200", 6)
                send_cmd(ser_motor, "m 3 0 6000", 4)

            return 0.0, dilution_index

    print("  Pre-detection passed; continuing the physical experiment.")

    if cycle % 5 != 0:

        send_cmd(ser_motor, "m 3 1 10000", 1)
        send_cmd(ser_motor, "m 2 1 9600", 2)
        send_cmd(ser_motor, "g 35", 4)
        send_cmd(ser_motor, "m 1 0 42400", 22)
        send_cmd(ser_motor, "m 3 0 10000", 9)
        send_cmd(ser_motor, "g 53", 1.5)
        send_cmd(ser_motor, "m 3 1 22000", 10)
        send_cmd(ser_motor, "m 1 1 36700", 18)

        if cycle % 5 == 1:
            send_cmd(ser_motor, "m 2 1 2000", 3.5)
            send_cmd(ser_motor, "m 3 0 12000", 10)
            send_cmd(ser_motor, "g 35", 2)
            send_cmd(ser_motor, "m 2 1 13500", 7.5)
            send_cmd(ser_motor, "m 3 0 8000", 5.5)
            send_cmd(ser_motor, "g 53", 2)
            send_cmd(ser_motor, "m 3 1 20000", 11)
            send_cmd(ser_motor, "m 2 0 15900", 8)
            send_cmd(ser_motor, "m 1 0 37000", 7)
            send_cmd(ser_motor, "m 3 0 16000", 12)
            send_cmd(ser_motor, "g 35", 2)
        elif cycle % 5 == 2:
            send_cmd(ser_motor, "m 2 1 15900", 8)
            send_cmd(ser_motor, "m 3 0 12000", 10)
            send_cmd(ser_motor, "g 35", 2)
            send_cmd(ser_motor, "m 2 1 10000", 6)
            send_cmd(ser_motor, "m 3 0 8000", 6.5)
            send_cmd(ser_motor, "g 53", 1)
            send_cmd(ser_motor, "m 3 1 20000", 13)
            send_cmd(ser_motor, "m 2 0 26300", 10)
            send_cmd(ser_motor, "m 1 0 37000", 13.5)
            send_cmd(ser_motor, "m 3 0 16000", 12.5)
            send_cmd(ser_motor, "g 35", 2)
        elif cycle % 5 == 3:
            send_cmd(ser_motor, "m 2 1 25800", 12)
            send_cmd(ser_motor, "m 3 0 12000", 10)
            send_cmd(ser_motor, "g 35", 2)
            send_cmd(ser_motor, "m 2 1 9800", 6.5)
            send_cmd(ser_motor, "m 3 0 8000", 6.5)
            send_cmd(ser_motor, "g 53", 1)
            send_cmd(ser_motor, "m 3 1 20000", 13)
            send_cmd(ser_motor, "m 2 0 36000", 15)
            send_cmd(ser_motor, "m 1 0 37000", 14)
            send_cmd(ser_motor, "m 3 0 16000", 12)
            send_cmd(ser_motor, "g 35", 1.5)
        elif cycle % 5 == 4:
            send_cmd(ser_motor, "m 2 1 35700", 15)
            send_cmd(ser_motor, "m 3 0 12000", 10)
            send_cmd(ser_motor, "g 35", 2)
            send_cmd(ser_motor, "m 2 1 9700", 6)
            send_cmd(ser_motor, "m 3 0 8000", 6.5)
            send_cmd(ser_motor, "g 53", 0.5)
            send_cmd(ser_motor, "m 3 1 20000", 13)
            send_cmd(ser_motor, "m 2 0 45800", 20)
            send_cmd(ser_motor, "m 1 0 37000", 14)
            send_cmd(ser_motor, "m 3 0 16000", 12.5)
            send_cmd(ser_motor, "g 35", 1.5)

    # ========================================================================
    # Stage 2: reagent dilution
    # ========================================================================
    print(f"\n--- Stage 2: Reagent dilution ---")

    final_water_well = target_ion_well
    if scw_value != 0.0:
        if scw_value == 0.005:
            mid = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            final_water_well = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            cmd = f"dilute_solute_{target_sw_well}_solvent_{target_ion_well}_ratio_0.005_mid_{mid}_target_{final_water_well}"
            send_cmd_wait_done(ser_ot2, cmd, timeout=300)
        elif scw_value == 0.001:
            m1 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            m2 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            final_water_well = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            cmd = f"dilute_solute_{target_sw_well}_solvent_{target_ion_well}_ratio_0.001_mid1_{m1}_mid2_{m2}_target_{final_water_well}"
            send_cmd_wait_done(ser_ot2, cmd, timeout=300)
        elif scw_value == 0.0002:
            m1 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            m2 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            final_water_well = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            cmd = f"dilute_solute_{target_sw_well}_solvent_{target_ion_well}_ratio_0.0002_mid1_{m1}_mid2_{m2}_target_{final_water_well}"
            send_cmd_wait_done(ser_ot2, cmd, timeout=300)

    final_oil_well = target_oil_well
    if sco_value != 0.0:
        if sco_value == 0.005:
            mid = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            final_oil_well = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            cmd = f"dilute_solute_{target_so_well}_solvent_{target_oil_well}_ratio_0.005_mid_{mid}_target_{final_oil_well}"
            send_cmd_wait_done(ser_ot2, cmd, timeout=300)
        elif sco_value == 0.001:
            m1 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            m2 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            final_oil_well = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            cmd = f"dilute_solute_{target_so_well}_solvent_{target_oil_well}_ratio_0.001_mid1_{m1}_mid2_{m2}_target_{final_oil_well}"
            send_cmd_wait_done(ser_ot2, cmd, timeout=300)
        elif sco_value == 0.0002:
            m1 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            m2 = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            final_oil_well = dilution_wells[dilution_index % len(dilution_wells)]; dilution_index += 1
            cmd = f"dilute_solute_{target_so_well}_solvent_{target_oil_well}_ratio_0.0002_mid1_{m1}_mid2_{m2}_target_{final_oil_well}"
            send_cmd_wait_done(ser_ot2, cmd, timeout=300)

    # ========================================================================
    # Stage 3: chip dispensing
    # ========================================================================
    print(f"\n--- Stage 3: Chip dispensing ---")
    send_cmd_wait_done(ser_ot2, f"water_{final_water_well}_home_to_1_left_&_no_home", timeout=300)
    send_cmd_wait_done(ser_ot2, f"oil_{final_oil_well}_home_to_1_right_&_no_home", timeout=300)
    send_cmd_wait_done(ser_ot2, "MNP_home_to_1_left_&_no_home", timeout=300)

    if cycle % 5 == 0:
        send_cmd(ser_motor, "m 3 1 10000", 1)
        send_cmd(ser_motor, "m 2 1 9600", 2)
        send_cmd(ser_motor, "g 35", 4)
        send_cmd(ser_motor, "m 1 0 42400", 22)
        send_cmd(ser_motor, "m 3 0 10000", 9)

    # ========================================================================
    # Stage 4: MNP transport and imaging
    # ========================================================================
    print(f"\n--- Stage 4: MNP transport and imaging ---")

    send_cmd(ser_motor, "m 2 1 37000", 12)
    send_cmd(ser_mnp, "R10000", 1)
    send_cmd(ser_motor, "m 3 1 15000", 4)
    send_cmd(ser_motor, "m 1 0 30000", 17)
    send_cmd(ser_motor, "g 82", 3)

    send_cmd(ser_motor, "m 3 1 9000", 6.5)
    send_cmd(ser_motor, "m 1 1 28500", 14)
    send_cmd(ser_motor, "m 3 0 14800", 12)
    send_cmd(ser_mnp, "L10000", 1)
    send_cmd(ser_motor, "g 5", 1.5)

    send_cmd(ser_motor, "m 3 1 6000", 4)
    send_cmd(ser_motor, "m 2 0 40500", 14)
    send_cmd(ser_motor, "m 1 0 31700", 14)
    send_cmd(ser_motor, "m 3 0 2500", 3.5)
    send_cmd(ser_motor, "g 25", 2)

    send_cmd(ser_motor, "m 3 1 14000", 2)
    send_cmd(ser_motor, "m 1 1 13600", 7)
    send_cmd(ser_motor, "m 2 1 18500", 9)
    send_cmd(ser_motor, "m 1 1 19200", 7)
    send_cmd(ser_motor, "m 2 1 12500", 8)

    # ========================================================================
    # Stage 5: CV scoring
    # ========================================================================
    print(f"\n--- Stage 5: CV droplet-quality scoring ---")
    time.sleep(1)
    reward = ask_chara_for_reward()

    try:
        df_update = pd.read_csv(csv_file)
        df_update.loc[row_index, 'reward'] = reward
        df_update.to_csv(csv_file, index=False)
        print(f"  reward={reward} written to {csv_file}, row {row_index}.")
    except Exception as e:
        print(f"  ERROR: failed to write CSV: {e}")

    # ========================================================================
    # Stage 6: camera reset and chip replacement
    # ========================================================================
    print(f"\n--- Stage 6: Camera reset and chip replacement ---")

    send_cmd(ser_motor, "m 2 0 12500", 5)
    send_cmd(ser_motor, "m 1 0 19200", 10)
    send_cmd(ser_motor, "m 2 0 18500", 9)
    send_cmd(ser_motor, "m 1 0 13600", 1)
    send_cmd(ser_motor, "m 3 0 14000", 11.5)
    send_cmd(ser_motor, "g 5", 1)

    if cycle % 5 == 1:
        send_cmd(ser_motor, "m 1 1 55000", 6)
        send_cmd(ser_motor, "m 3 0 5000", 12)
        send_cmd(ser_motor, "g 35", 1)
        send_cmd(ser_motor, "m 2 1 53500", 27)
        send_cmd(ser_motor, "m 3 0 10000", 9)
        send_cmd(ser_motor, "g 82", 2)
        send_cmd(ser_motor, "m 3 1 25000", 14)
        send_cmd(ser_motor, "m 2 1 4500", 4)
        send_cmd(ser_motor, "m 1 0 52000", 18)
        send_cmd(ser_motor, "m 2 0 17700", 10)
        send_cmd(ser_motor, "m 3 0 6800", 7)
        send_cmd(ser_motor, "g 35", 2)
        send_cmd(ser_motor, "m 1 1 29000", 8)
        send_cmd(ser_motor, "m 3 0 9000", 7.5)
        send_cmd(ser_motor, "g 82", 2)
        send_cmd(ser_motor, "m 3 1 15500", 10.5)
        send_cmd(ser_motor, "m 1 1 23500", 4)
        send_cmd(ser_motor, "m 2 1 12800", 7)
        send_cmd(ser_motor, "m 3 0 25000", 19)
        send_cmd(ser_motor, "g 35", 2)
        send_cmd(ser_motor, "m 1 1 20400", 12)
        send_cmd(ser_motor, "m 2 0 58800", 16)
        send_cmd(ser_motor, "m 3 0 3400", 7)

    elif cycle % 5 == 2:
        send_cmd(ser_motor, "m 1 1 55000", 6)
        send_cmd(ser_motor, "m 3 0 5000", 12)
        send_cmd(ser_motor, "g 35", 1)
        send_cmd(ser_motor, "m 2 1 34500", 18)
        send_cmd(ser_motor, "m 3 0 10000", 8)
        send_cmd(ser_motor, "g 82", 2)
        send_cmd(ser_motor, "m 3 1 26000", 15)
        send_cmd(ser_motor, "m 2 1 23500", 13)
        send_cmd(ser_motor, "m 1 0 52100", 17)
        send_cmd(ser_motor, "m 3 0 2000", 1)
        send_cmd(ser_motor, "m 2 0 17700", 10)
        send_cmd(ser_motor, "m 3 0 5800", 6.5)
        send_cmd(ser_motor, "g 35", 1.5)
        send_cmd(ser_motor, "m 1 1 29100", 8)
        send_cmd(ser_motor, "m 3 0 9000", 7.5)
        send_cmd(ser_motor, "g 82", 1.5)
        send_cmd(ser_motor, "m 3 1 15500", 10.5)
        send_cmd(ser_motor, "m 1 1 23500", 4)
        send_cmd(ser_motor, "m 2 0 6000", 5)
        send_cmd(ser_motor, "m 3 0 25000", 19)
        send_cmd(ser_motor, "g 35", 2)
        send_cmd(ser_motor, "m 1 1 20400", 12)
        send_cmd(ser_motor, "m 2 0 40000", 16)
        send_cmd(ser_motor, "m 3 0 3400", 7)

    elif cycle % 5 == 3:
        send_cmd(ser_motor, "m 1 1 55000", 6)
        send_cmd(ser_motor, "m 3 0 5000", 12)
        send_cmd(ser_motor, "g 35", 1)
        send_cmd(ser_motor, "m 2 1 10500", 16)
        send_cmd(ser_motor, "m 3 0 10000", 8)
        send_cmd(ser_motor, "g 82", 2)
        send_cmd(ser_motor, "m 3 1 26000", 2)
        send_cmd(ser_motor, "m 2 1 47500", 20)
        send_cmd(ser_motor, "m 1 0 52200", 17)
        send_cmd(ser_motor, "m 3 0 2000", 1)
        send_cmd(ser_motor, "m 2 0 17700", 10)
        send_cmd(ser_motor, "m 3 0 5800", 6)
        send_cmd(ser_motor, "g 35", 1.5)
        send_cmd(ser_motor, "m 1 1 29200", 8)
        send_cmd(ser_motor, "m 3 0 9000", 7.5)
        send_cmd(ser_motor, "g 82", 2)
        send_cmd(ser_motor, "m 3 1 15500", 10.5)
        send_cmd(ser_motor, "m 1 1 23500", 4)
        send_cmd(ser_motor, "m 2 0 29800", 3)
        send_cmd(ser_motor, "m 3 0 25000", 19)
        send_cmd(ser_motor, "g 35", 2)
        send_cmd(ser_motor, "m 1 1 20400", 12)
        send_cmd(ser_motor, "m 2 0 16200", 10)
        send_cmd(ser_motor, "m 3 0 3400", 7)

    elif cycle % 5 == 4:
        send_cmd(ser_motor, "m 1 1 55000", 6)
        send_cmd(ser_motor, "m 3 0 5000", 12)
        send_cmd(ser_motor, "g 35", 1)
        send_cmd(ser_motor, "m 2 0 4500", 8)
        send_cmd(ser_motor, "m 3 0 10000", 9)
        send_cmd(ser_motor, "g 82", 2)
        send_cmd(ser_motor, "m 3 1 26000", 1)
        send_cmd(ser_motor, "m 2 1 62500", 30)
        send_cmd(ser_motor, "m 1 0 52200", 17)
        send_cmd(ser_motor, "m 3 0 2000", 1)
        send_cmd(ser_motor, "m 2 0 17900", 10)
        send_cmd(ser_motor, "m 3 0 5800", 6.5)
        send_cmd(ser_motor, "g 35", 1.5)
        send_cmd(ser_motor, "m 1 1 29200", 8)
        send_cmd(ser_motor, "m 3 0 9000", 7.5)
        send_cmd(ser_motor, "g 82", 1)
        send_cmd(ser_motor, "m 3 1 15500", 10.5)
        send_cmd(ser_motor, "m 1 1 23500", 4)
        send_cmd(ser_motor, "m 2 0 45000", 22)
        send_cmd(ser_motor, "m 3 0 25000", 19)
        send_cmd(ser_motor, "g 35", 2)
        send_cmd(ser_motor, "m 1 1 20400", 12)
        send_cmd(ser_motor, "m 2 0 800", 1)
        send_cmd(ser_motor, "m 3 0 3400", 7)

    elif cycle % 5 == 0:
        send_cmd(ser_motor, "m 1 1 75600", 22)
        send_cmd(ser_motor, "m 2 0 6100", 7)
        send_cmd(ser_motor, "g 90", 1)
        send_cmd(ser_motor, "m 3 0 12700", 2)

    print(f"\nExperiment cycle {cycle} completed. Reward = {reward:.4f}")

    if cycle % 5 == 0:
        input("\nPlease manually replace or refill consumables, then press Enter to continue...\n")

    return reward, dilution_index


# ============================================================================
# 6. Main loop
# ============================================================================

def main():
    print("\n" + "="*60)
    print("   A3OF Master Closed-Loop Controller")
    print("   Fully autonomous droplet microfluidic Bayesian optimization loop")
    print("="*60)

    if not os.path.exists(CSV_FILE):
        print(f"ERROR: cannot find {CSV_FILE}. Please make sure the data file exists.")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} historical experiment records.")

    connect_hardware()

    for sig_file in [PRE_DETECT_SIGNAL, CHARA_SIGNAL]:
        if not os.path.exists(sig_file):
            with open(sig_file, 'w') as f:
                f.write("WAITING")

    dilution_index = 0
    cycle = 1

    print("\n" + "="*60)
    print("   Closed-loop optimization started.")
    print("="*60)

    try:
        while cycle <= MAX_ITERATIONS:
            print(f"\n{'-'*40}")
            print(f"Cycle {cycle}: BO is analyzing data and recommending a recipe...")

            if BOFIRE_AVAILABLE:
                new_exp = bo_recommend(CSV_FILE)
            else:
                new_exp = random_recommend()

            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, new_exp], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)

            new_row_index = len(df) - 1
            last_row = df.iloc[-1]
            params = {
                "SW":  str(last_row["SW"]).strip(),
                "SCW": float(last_row["Rwater"]),
                "SCO": float(last_row["ROil"]),
                "OT":  str(last_row["Oil"]).strip(),
                "SO":  str(last_row["SO"]).strip(),
                "Ion": float(last_row["Ion"]),
            }

            reward, dilution_index = execute_single_experiment(
                params, cycle, CSV_FILE, new_row_index, dilution_index
            )

            cycle += 1

    except KeyboardInterrupt:
        print("\n\nUser interrupted the run. Exiting safely...")
        emit_stop()
    except Exception as e:
        print(f"\nERROR: runtime exception: {e}")
        traceback.print_exc()
        emit_stop()
    finally:
        disconnect_hardware()
        print(f"\nClosed-loop run ended after {cycle - 1} experiment cycles.")
        print(f"   Data saved to: {CSV_FILE}")


# ============================================================================
# 7. Entrypoint
# ============================================================================
if __name__ == "__main__":
    main()
