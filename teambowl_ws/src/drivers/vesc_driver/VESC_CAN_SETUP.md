# VESC CAN Bus Setup Guide

Configure each VESC via VESC Tool before running the updated driver.

## Requirements

- VESC Tool desktop app (download from vesc-project.com)
- USB-C cable to connect each VESC individually to a laptop
- Do this for **each** VESC separately

## Steps (repeat for both VESCs)

### 1. Connect

Plug the VESC into your laptop via USB. Power the VESC from its normal supply (battery or bench supply).  
Open VESC Tool → click **AutoConnect** (or manually select the serial port).

### 2. Set the CAN ID

Left sidebar → **App Settings** → **General**

Set **Controller ID** to:
- `14` — left wheel motor
- `24` — right wheel motor

Click **Write**.

### 3. Set CAN baud rate

Still in **App Settings** → **General**

Set **CAN Baud Rate** to **1000 kbps** (must match `can1` on the Jetson).

Click **Write**.

### 4. Enable CAN status messages

Still in **App Settings** → **General**, scroll to the **CAN Status Message** section:

- Set **CAN Status Message Mode** to **Status 1 and 5** (or "All" if available)
- Set **CAN Status Rate** to **50 Hz** (or highest available ≥ 20 Hz)

Click **Write**.

> Status 1 carries RPM, current, and duty cycle.  
> Status 5 carries tachometer and input voltage (battery voltage).

### 5. Configure command timeout

Still in **App Settings** → **General**:

- Set **Timeout** to `500` ms — the VESC will coast if no CAN frame arrives for 500 ms
- Set **Timeout Brake Current** to `0.0` A — timeout results in free coast, not braking

Click **Write**.

### 6. Save to EEPROM

- Top toolbar → **Motor Settings** → click **Write Motor Configuration**
- Top toolbar → **App Settings** → click **Write App Configuration**

Settings now survive power cycles.

### 7. Verify on the Jetson

With `can1` up on the Jetson, power-cycle the VESC and run:

```bash
candump can1
```

You should see periodic auto-broadcast frames:

| VESC  | Status 1 frame ID | Status 5 frame ID |
|-------|------------------|------------------|
| Left  | 0x090E           | 0x1B0E           |
| Right | 0x0918           | 0x1B18           |

## Quick Reference

| Parameter         | Left VESC | Right VESC |
|-------------------|-----------|------------|
| Controller ID     | 14        | 24         |
| CAN Baud Rate     | 1000 kbps | 1000 kbps  |
| Status Mode       | 1 and 5   | 1 and 5    |
| Status Rate       | 50 Hz     | 50 Hz      |
| Timeout           | 500 ms    | 500 ms     |
| Timeout Brake     | 0.0 A     | 0.0 A      |

## Why SET_CURRENT(0) for coast

The driver sends `CAN_PACKET_SET_CURRENT` with 0 mA for stop/coast instead of `SET_DUTY(0)`.  
`SET_DUTY(0)` tries to maintain 0% duty cycle and causes regenerative braking.  
`SET_CURRENT(0)` commands zero torque — the motor windings are released and the wheel spins freely.  
The VESC timeout (step 5 above) also ensures coast if the Jetson crashes or the CAN cable is disconnected.
