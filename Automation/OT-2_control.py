"""
OT-2 liquid-handling control script for A3OF.

Run this on the OT-2 Raspberry Pi. It receives commands forwarded by ESP32
over serial, executes pipetting, dilution, and dispensing actions, and returns
"DONE_OT2" after each action.
"""
import opentrons.execute
from opentrons import protocol_api, types
import serial
import time
import glob
import re

print("\n============================================")
print("[System startup] A3OF OT-2 automation script v2.0")
print("============================================")

# =====================================================================
# OT-2 setup
# =====================================================================
print("[1/4] Loading OT-2 protocol API 2.19. This may take 30-50 seconds...")
protocol = opentrons.execute.get_protocol_api("2.19")

print("[2/4] Homing all axes. The gantry will move over a large range.")
protocol.home()
print("Homing complete. Stepper motors are ready.")

print("[3/4] Mapping labware and pipette...")
plate_1 = protocol.load_labware("corning_96_wellplate_360ul_flat", location="1")
plate_3 = protocol.load_labware("corning_96_wellplate_360ul_flat", location="3")
plate_4 = protocol.load_labware("corning_96_wellplate_360ul_flat", location="4")
plate_5 = protocol.load_labware("corning_96_wellplate_360ul_flat", location="5")
plate_6 = protocol.load_labware("corning_96_wellplate_360ul_flat", location="6")

tiprack_9 = protocol.load_labware("opentrons_96_tiprack_20ul", location="9")

left_pipette = protocol.load_instrument(
    "p20_single_gen2", mount="left", tip_racks=[tiprack_9]
)
left_pipette.starting_tip = tiprack_9["A9"]
print("Labware and pipette mapping complete. Tips will start from A9.")

# =====================================================================
# Serial setup
# =====================================================================
print("[4/4] Searching for the ESP32 connected to OT-2...")

possible_ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
ser = None

if not possible_ports:
    print("FATAL: no USB serial device was detected. Check that the ESP32 is connected.")
else:
    print(f"Detected serial ports: {possible_ports}")
    for port in possible_ports:
        try:
            print(f"   -> Trying {port} ...")
            ser = serial.Serial(port, baudrate=115200, timeout=1)
            print(f"Connected to ESP32 on {port}.")
            break
        except Exception as e:
            print(f"   -> Failed to connect to {port}: {e}")
            ser = None

if ser is None:
    raise RuntimeError("Serial connection failed. Reconnect the ESP32 and try again.")

print("\n[Ready] OT-2 is listening for wireless commands from the main controller...")

# =====================================================================
# Dilution recipes for the P20 pipette, with volumes in microliters.
# 0.005: 2 uL stock + 18 uL solvent, then 2 uL mid + 38 uL solvent.
# 0.001: 2 uL stock + 18 uL solvent, then 2 uL mid1 + 18 uL solvent,
#        then 4 uL mid2 + 36 uL solvent.
# 0.0002: 2 uL stock + 18 uL solvent, then 1 uL mid1 + 19 uL solvent,
#         then 1.6 uL mid2 + 38.4 uL solvent.
# =====================================================================

def perform_action(cmd):
    print(f"\nReceived command: [{cmd}]. Starting action...")

    match_water = re.match(r"water_(.+)_home_to_1_left_&_(yes|no)_home", cmd)
    match_oil = re.match(r"oil_(.+)_home_to_1_right_&_(yes|no)_home", cmd)
    match_MNP = re.match(r"MNP_home_to_1_left_&_(yes|no)_home", cmd)
    match_shiguan_oil = re.match(r"(.+)_shiguan_oil_&_(yes|no)_home", cmd)
    match_shiguan_surfactant = re.match(r"(.+)_shiguan_surfactant_&_(yes|no)_home", cmd)

    match_dilute_2step = re.match(
        r"dilute_solute_(.+?)_solvent_(.+?)_ratio_([\d\.]+)_mid_(.+?)_target_(.+)", cmd
    )
    match_dilute_3step = re.match(
        r"dilute_solute_(.+?)_solvent_(.+?)_ratio_([\d\.]+)_mid1_(.+?)_mid2_(.+?)_target_(.+)", cmd
    )

    if cmd == "home":
        protocol.home()
        print("Action complete: all axes homed.")

    elif cmd == "pick":
        left_pipette.pick_up_tip()
        print("Action complete: tip picked up.")

    elif cmd == "drop":
        left_pipette.drop_tip()
        print("Action complete: tip dropped.")

    elif match_dilute_2step:
        solute = match_dilute_2step.group(1)
        solvent = match_dilute_2step.group(2)
        ratio = float(match_dilute_2step.group(3))
        mid_loc = match_dilute_2step.group(4)
        target_loc = match_dilute_2step.group(5)

        print(f"Two-step dilution for ratio {ratio}: solute={solute}, solvent={solvent}, mid={mid_loc}, target={target_loc}")

        if ratio == 0.005:
            vol_s1_solute, vol_s1_solvent = 2.0, 18.0
            vol_s2_mid, vol_s2_solvent = 2.0, 38.0

            left_pipette.pick_up_tip()
            left_pipette.transfer(vol_s1_solvent, plate_6[solvent].bottom(1),
                                  plate_6[mid_loc].bottom(1), new_tip='never')
            left_pipette.blow_out(plate_6[mid_loc].bottom(5))
            left_pipette.drop_tip()

            left_pipette.pick_up_tip()
            left_pipette.transfer(vol_s1_solute, plate_6[solute].bottom(1),
                                  plate_6[mid_loc].bottom(1), new_tip='never')
            left_pipette.mix(2, 15, plate_6[mid_loc].bottom(1))
            left_pipette.blow_out(plate_6[mid_loc].bottom(5))
            left_pipette.drop_tip()
            print(f"Step 1 complete: {vol_s1_solute+vol_s1_solvent} uL intermediate solution at {mid_loc}, ratio 0.1.")

            left_pipette.pick_up_tip()
            left_pipette.transfer(vol_s2_solvent, plate_6[solvent].bottom(1),
                                  plate_6[target_loc].bottom(1), new_tip='never')
            left_pipette.blow_out(plate_6[target_loc].bottom(5))
            left_pipette.drop_tip()

            left_pipette.pick_up_tip()
            left_pipette.transfer(vol_s2_mid, plate_6[mid_loc].bottom(1),
                                  plate_6[target_loc].bottom(1), new_tip='never')
            left_pipette.mix(2, 15, plate_6[target_loc].bottom(1))
            left_pipette.blow_out(plate_6[target_loc].bottom(5))
            left_pipette.drop_tip()
            print(f"Step 2 complete: {vol_s2_mid+vol_s2_solvent} uL final solution at {target_loc}, ratio {ratio}.")

        else:
            print(f"Warning: two-step dilution does not support ratio {ratio}. Check the command.")

    elif match_dilute_3step:
        solute = match_dilute_3step.group(1)
        solvent = match_dilute_3step.group(2)
        ratio = float(match_dilute_3step.group(3))
        mid1_loc = match_dilute_3step.group(4)
        mid2_loc = match_dilute_3step.group(5)
        target_loc = match_dilute_3step.group(6)

        print(f"Three-step dilution for ratio {ratio}: solute={solute}, solvent={solvent}, mid1={mid1_loc}, mid2={mid2_loc}, target={target_loc}")

        if ratio == 0.001:
            s1_solute, s1_solvent = 2.0, 18.0
            s2_mid1, s2_solvent = 2.0, 18.0
            s3_mid2, s3_solvent = 4.0, 36.0
        elif ratio == 0.0002:
            s1_solute, s1_solvent = 2.0, 18.0
            s2_mid1, s2_solvent = 1.0, 19.0
            s3_mid2, s3_solvent = 1.6, 38.4
        else:
            print(f"Warning: three-step dilution does not support ratio {ratio}. Check the command.")
            return

        left_pipette.pick_up_tip()
        left_pipette.transfer(s1_solvent, plate_6[solvent].bottom(1),
                              plate_6[mid1_loc].bottom(1), new_tip='never')
        left_pipette.blow_out(plate_6[mid1_loc].bottom(5))
        left_pipette.drop_tip()

        left_pipette.pick_up_tip()
        left_pipette.transfer(s1_solute, plate_6[solute].bottom(1),
                              plate_6[mid1_loc].bottom(1), new_tip='never')
        left_pipette.mix(2, 15, plate_6[mid1_loc].bottom(1))
        left_pipette.blow_out(plate_6[mid1_loc].bottom(5))
        left_pipette.drop_tip()
        print(f"Step 1 complete: {s1_solute+s1_solvent} uL at {mid1_loc}.")

        left_pipette.pick_up_tip()
        left_pipette.transfer(s2_solvent, plate_6[solvent].bottom(1),
                              plate_6[mid2_loc].bottom(1), new_tip='never')
        left_pipette.blow_out(plate_6[mid2_loc].bottom(5))
        left_pipette.drop_tip()

        left_pipette.pick_up_tip()
        left_pipette.transfer(s2_mid1, plate_6[mid1_loc].bottom(1),
                              plate_6[mid2_loc].bottom(1), new_tip='never')
        left_pipette.mix(2, 15, plate_6[mid2_loc].bottom(1))
        left_pipette.blow_out(plate_6[mid2_loc].bottom(5))
        left_pipette.drop_tip()
        print(f"Step 2 complete: {s2_mid1+s2_solvent} uL at {mid2_loc}.")

        left_pipette.pick_up_tip()
        left_pipette.transfer(s3_solvent, plate_6[solvent].bottom(1),
                              plate_6[target_loc].bottom(1), new_tip='never')
        left_pipette.blow_out(plate_6[target_loc].bottom(5))
        left_pipette.drop_tip()

        left_pipette.pick_up_tip()
        left_pipette.transfer(s3_mid2, plate_6[mid2_loc].bottom(1),
                              plate_6[target_loc].bottom(1), new_tip='never')
        left_pipette.mix(2, 15, plate_6[target_loc].bottom(1))
        left_pipette.blow_out(plate_6[target_loc].bottom(5))
        left_pipette.drop_tip()
        print(f"Step 3 complete: {s3_mid2+s3_solvent} uL final solution at {target_loc}, ratio {ratio}.")

    elif match_water:
        well_name = match_water.group(1)
        need_home = match_water.group(2)

        left_pipette.pick_up_tip()
        asp_loc = plate_6[well_name].bottom(1)
        left_pipette.aspirate(20, asp_loc)
        print(f"Action complete: aspirated water-phase liquid from well {well_name}.")

        safe_up_loc = plate_6["H10"].top().move(types.Point(x=0, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_up_loc = plate_4["H1"].top().move(types.Point(x=50, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_across_loc = plate_1["E6"].top().move(types.Point(x=2, y=4, z=134))
        left_pipette.move_to(safe_across_loc)
        disp_loc = plate_1["E6"].top().move(types.Point(x=2, y=4, z=45))
        left_pipette.dispense(20, disp_loc)
        left_pipette.blow_out(disp_loc)
        print("Action complete: dispensed into the left chip inlet.")

        safe_across_loc = plate_1["E6"].top().move(types.Point(x=2, y=4, z=134))
        left_pipette.move_to(safe_across_loc)
        safe_up_loc = plate_4["H1"].top().move(types.Point(x=50, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_up_loc = plate_6["H12"].top().move(types.Point(x=0, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)

        if need_home == "yes":
            protocol.home()
        left_pipette.drop_tip()
        print(f"Action complete: {match_water.group(0)}")

    elif match_oil:
        well_name = match_oil.group(1)
        need_home = match_oil.group(2)

        left_pipette.pick_up_tip()
        asp_loc = plate_6[well_name].bottom(1)
        left_pipette.aspirate(20, asp_loc)
        print(f"Action complete: aspirated oil-phase liquid from well {well_name}.")

        safe_up_loc = plate_6["H10"].top().move(types.Point(x=0, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_up_loc = plate_4["H1"].top().move(types.Point(x=63, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_across_loc = plate_1["E6"].top().move(types.Point(x=14, y=4, z=134))
        left_pipette.move_to(safe_across_loc)
        disp_loc = plate_1["E6"].top().move(types.Point(x=14, y=4, z=45))
        left_pipette.dispense(20, disp_loc)
        left_pipette.blow_out(disp_loc)
        print("Action complete: dispensed into the right chip inlet.")

        safe_across_loc = plate_1["E6"].top().move(types.Point(x=14, y=4, z=134))
        left_pipette.move_to(safe_across_loc)
        safe_up_loc = plate_4["H1"].top().move(types.Point(x=63, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_up_loc = plate_6["H12"].top().move(types.Point(x=0, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)

        if need_home == "yes":
            protocol.home()
        left_pipette.drop_tip()
        print(f"Action complete: {match_oil.group(0)}")

    elif match_MNP:
        need_home = match_MNP.group(1)

        left_pipette.pick_up_tip()
        asp_loc = plate_6["H5"].bottom(1)
        left_pipette.aspirate(1, asp_loc)
        print("Action complete: aspirated MNP from well H5.")

        safe_up_loc = plate_6["H5"].top().move(types.Point(x=0, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_up_loc = plate_4["H1"].top().move(types.Point(x=50, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_across_loc = plate_1["E6"].top().move(types.Point(x=2, y=4, z=134))
        left_pipette.move_to(safe_across_loc)
        disp_loc = plate_1["E6"].top().move(types.Point(x=2, y=4, z=45))
        left_pipette.dispense(1, disp_loc)
        left_pipette.blow_out(disp_loc)
        print("Action complete: dispensed MNP into the left chip inlet.")

        safe_across_loc = plate_1["E6"].top().move(types.Point(x=2, y=4, z=134))
        left_pipette.move_to(safe_across_loc)
        safe_up_loc = plate_4["H1"].top().move(types.Point(x=50, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)
        safe_up_loc = plate_6["H12"].top().move(types.Point(x=0, y=-25, z=134))
        left_pipette.move_to(safe_up_loc)

        if need_home == "yes":
            protocol.home()
        left_pipette.drop_tip()
        print(f"Action complete: {match_MNP.group(0)}")

    elif match_shiguan_oil:
        well_name = match_shiguan_oil.group(1)
        need_home = match_shiguan_oil.group(2)

        left_pipette.pick_up_tip()
        asp_loc = plate_6[well_name].bottom(1)
        left_pipette.aspirate(20, asp_loc)
        print(f"Action complete: aspirated liquid from well {well_name}.")

        disp_loc = plate_5["D2"].top().move(types.Point(x=2, y=-4, z=35))
        left_pipette.dispense(20, disp_loc)
        left_pipette.blow_out(disp_loc)
        print("Action complete: dispensed into the pre-detection tube.")

        safe_up_loc = plate_5["D2"].top().move(types.Point(x=2, y=-4, z=115))
        left_pipette.move_to(safe_up_loc)
        safe_across_loc = plate_6["D12"].top().move(types.Point(x=2, y=-4, z=115))
        left_pipette.move_to(safe_across_loc)

        if need_home == "yes":
            protocol.home()
        left_pipette.drop_tip()
        print(f"Action complete: {match_shiguan_oil.group(0)}")

    elif match_shiguan_surfactant:
        well_name = match_shiguan_surfactant.group(1)
        need_home = match_shiguan_surfactant.group(2)

        left_pipette.pick_up_tip()
        asp_loc = plate_6[well_name].bottom(1)
        left_pipette.aspirate(20, asp_loc)
        print(f"Action complete: aspirated liquid from well {well_name}.")

        disp_loc = plate_5["D2"].top().move(types.Point(x=2, y=-4, z=35))
        left_pipette.dispense(20, disp_loc)
        left_pipette.blow_out(disp_loc)
        print("Action complete: dispensed into the pre-detection tube.")

        safe_up_loc = plate_5["D2"].top().move(types.Point(x=2, y=-4, z=115))
        left_pipette.move_to(safe_up_loc)
        safe_across_loc = plate_6["D12"].top().move(types.Point(x=2, y=-4, z=115))
        left_pipette.move_to(safe_across_loc)

        if need_home == "yes":
            protocol.home()
        left_pipette.drop_tip()
        print(f"Action complete: {well_name}")

    else:
        print(f"Warning: unknown command [{cmd}]. The robot will stay idle.")


# =====================================================================
# Main serial command loop
# =====================================================================
while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()

        prefix = "TO PC >>> "
        if line.startswith(prefix):
            actual_cmd = line[len(prefix):]
            perform_action(actual_cmd)

            ser.write(b"DONE_OT2\n")

    time.sleep(0.05)
