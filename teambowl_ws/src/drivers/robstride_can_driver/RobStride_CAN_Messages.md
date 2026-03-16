# RobStride Motor CAN Message Reference

**Covers:** RS00 · RS04 · RS05  
**Bus:** CAN 2.0, 1 Mbps, Extended Frame (29-bit ID)  
**Protocols supported:** Private (default) · CANopen · MIT

---

## Frame Structure

All private-protocol messages use a 29-bit extended CAN ID split into three fields, plus an 8-byte data area:

| Field | Bits | Description |
|---|---|---|
| Communication Type | Bit28~Bit24 | Message type identifier |
| Data Area 2 | Bit23~Bit8 | Host CAN ID (Bit15~8) and/or secondary data |
| Destination Address | Bit7~Bit0 | Target motor CAN ID |
| Data Area 1 | Byte0~Byte7 | Payload |

---

## Private Protocol — Communication Types

---

### Type 0 — Get Device ID (`0x0`)

Requests the motor's 64-bit MCU unique identifier.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x0` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0~7 | `0x00 00 00 00 00 00 00 00` |

**Reply Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x0` |
| Bit23~8 | Target motor CAN ID |
| Bit7~0 | `0xFE` |
| Byte0~7 | 64-bit MCU unique identifier |

---

### Type 1 — Operation Control Mode Motor Control (`0x1`)

Sends all five MIT-style motion control parameters in one frame. Motor must be enabled (Type 3) first.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x1` |
| Bit23~16 | Torque feedforward `[0~65535]` |
| Bit7~0 | Target motor CAN ID |
| Byte0~1 | Target angle `[0~65535]` → mapped to position range |
| Byte2~3 | Target angular velocity `[0~65535]` → mapped to velocity range |
| Byte4~5 | Kp `[0~65535]` → mapped to Kp range |
| Byte6~7 | Kd `[0~65535]` → mapped to Kd range |

> High byte is transmitted first (big-endian) for all multi-byte values.

**Motor-specific ranges for Type 1:**

| Parameter | RS00 | RS04 | RS05 |
|---|---|---|---|
| Torque (Byte2 in ID) | −14 Nm ~ +14 Nm | −120 Nm ~ +120 Nm | −5.5 Nm ~ +5.5 Nm |
| Target angle (Byte0~1) | −4π ~ +4π rad | −4π ~ +4π rad | −4π ~ +4π rad |
| Target velocity (Byte2~3) | −33 ~ +33 rad/s | −15 ~ +15 rad/s | −50 ~ +50 rad/s |
| Kp (Byte4~5) | 0 ~ 500 | 0 ~ 5000 | 0 ~ 500 |
| Kd (Byte6~7) | 0 ~ 5 | 0 ~ 100 | 0 ~ 5 |

**Response:** Type 2 motor feedback frame.

---

### Type 2 — Motor Feedback Data (`0x2`)

Sent by the motor in response to most commands, or periodically if active reporting (Type 24) is enabled.

**Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x2` |
| Bit15~8 | Motor CAN ID |
| Bit23~22 | Mode status: `0`=Reset, `1`=Cali, `2`=Motor (Run) |
| Bit21 | Fault: Uncalibrated |
| Bit20 | Fault: Gridlock/overload |
| Bit19 | Fault: Magnetic encoder fault |
| Bit18 | Fault: Overtemperature |
| Bit17 | Fault: Three-phase overcurrent |
| Bit16 | Fault: Undervoltage |
| Bit7~0 | Host CAN ID |
| Byte0~1 | Current angle `[0~65535]` → −4π ~ +4π rad |
| Byte2~3 | Current angular velocity `[0~65535]` → velocity range |
| Byte4~5 | Current torque `[0~65535]` → torque range |
| Byte6~7 | Winding temperature: `Temp(°C) × 10`, high byte first if value > 10 |

**Motor-specific ranges for Type 2:**

| Parameter | RS00 | RS04 | RS05 |
|---|---|---|---|
| Angular velocity | −33 ~ +33 rad/s | −15 ~ +15 rad/s | −50 ~ +50 rad/s |
| Torque | −14 ~ +14 Nm | −120 ~ +120 Nm | −5.5 ~ +5.5 Nm |

---

### Type 3 — Motor Enable (`0x3`)

Enables the motor for operation.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x3` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0~7 | (empty / zeroed) |

**Response:** Type 2 motor feedback frame.

---

### Type 4 — Motor Stop (`0x4`)

Stops the motor. Can also clear a fault on the same frame.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x4` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0 | `0x00` = normal stop; `0x01` = clear fault and stop |
| Byte1~7 | `0x00` (must be zero during normal operation) |

**Response:** Type 2 motor feedback frame.

---

### Type 6 — Set Motor Mechanical Zero (`0x6`)

Sets the current motor position as the new mechanical zero.  
> Not available in PP (Position Profile) mode.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x6` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0 | `0x01` |
| Byte1~7 | (unused) |

**Response:** Type 2 motor feedback frame.

---

### Type 7 — Set Motor CAN ID (`0x7`)

Changes the motor's CAN ID immediately (no reboot required).

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x7` |
| Bit23~16 | New (preset) CAN ID |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Current target motor CAN ID |
| Byte0~7 | (unused) |

**Response:** Type 0 broadcast reply frame.

---

### Type 17 — Single Parameter Read (`0x11`)

Reads one parameter from the motor's parameter table using its index.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x11` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0~1 | Parameter index (little-endian; see parameter table below) |
| Byte2~3 | `0x00 0x00` |
| Byte4~7 | `0x00 00 00 00` |

**Reply Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x11` |
| Bit23~16 | `0x00` = read success; `0x01` = error |
| Bit15~8 | Motor CAN ID |
| Bit7~0 | Host CAN ID |
| Byte0~1 | Parameter index (echo) |
| Byte2~3 | `0x00 0x00` |
| Byte4~7 | Parameter value (little-endian, IEEE-754 float or integer) |

**Example** — Reading `loc_kp` (index `0x701E`) from motor CAN ID `0x7F`, host `0xFD`:

```
TX: ID=0x1100FD7F  Data: 1E 70 00 00 00 00 00 00
RX: ID=0x11007FFD  Data: 1E 70 00 00 00 00 F0 41
```

---

### Type 18 — Single Parameter Write, Volatile (`0x12`)

Writes one parameter. Change is lost on power-off unless saved with Type 22.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x12` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0~1 | Parameter index (little-endian) |
| Byte2~3 | `0x00 0x00` |
| Byte4~7 | New value (little-endian, IEEE-754 float or integer) |

**Response:** Type 2 motor feedback frame.

---

### Type 21 — Fault Feedback Frame (`0x15`)

Sent by the motor to report active faults and warnings.

**Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x15` |
| Bit15~8 | Motor CAN ID |
| Bit7~0 | Host CAN ID |
| Byte0~3 | Fault value (non-zero = fault present) |
| Byte4~7 | Warning value |

**Fault bit map (Byte0~3):**

| Bit | Fault |
|---|---|
| Bit16 | A-phase current sampling overcurrent |
| Bit14 | Motor stall / I²t overload protection |
| Bit9 | Position initialization fault |
| Bit8 | Hardware identification fault |
| Bit7 | Encoder uncalibrated |
| Bit5 | C-phase current sampling overcurrent |
| Bit4 | B-phase current sampling overcurrent |
| Bit3 | Overvoltage fault |
| Bit2 | Undervoltage fault |
| Bit1 | Driver chip fault |
| Bit0 | Motor overtemperature (default threshold: RS00/RS05 = 135°C, RS04 = 145°C) |

**Warning bit map (Byte4~7):**

| Bit | Warning |
|---|---|
| Bit0 | Motor overtemperature warning (default 135°C) |

---

### Type 22 — Motor Data Save (`0x16`)

Persists all parameters modified via Type 18 to non-volatile storage.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x16` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0~7 | `01 02 03 04 05 06 07 08` |

**Response:** Type 2 motor feedback frame.

---

### Type 23 — Motor Baud Rate Modification (`0x17`)

Changes the CAN baud rate. Takes effect after power cycle.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x17` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte6 (F_CMD) | `0x01` = 1M, `0x02` = 500K, `0x03` = 250K, `0x04` = 125K |
| Byte0~5 | `01 02 03 04 05 06` (fixed padding) |

**Response:** Type 0 broadcast reply frame.

---

### Type 24 — Motor Active Reporting Control (`0x18`)

Enables or disables periodic unsolicited Type 2 feedback from the motor.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x18` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte6 (F_CMD) | `0x00` = disable (default); `0x01` = enable (default interval 10 ms) |
| Byte0~5 | `01 02 03 04 05 06` (fixed padding) |

> Adjust the reporting interval by writing `EPScan_time` (index `0x7026`) via Type 18. Value of 1 = 10 ms; each additional count adds 5 ms.

**Active Report Response Frame** (same structure as Type 2):

| Field | Value |
|---|---|
| Bit28~24 | `0x18` |
| Bit15~8 | Motor CAN ID |
| Bit23~22 | Mode status |
| Bit21~16 | Fault flags (same as Type 2) |
| Bit7~0 | Target/host CAN ID |
| Byte0~1 | Current angle `[0~65535]` → −4π ~ +4π rad |
| Byte2~3 | Current angular velocity → velocity range |
| Byte4~5 | Current torque → torque range |
| Byte6~7 | Current temperature: `Temp(°C) × 10` |

---

### Type 25 — Motor Protocol Modification (`0x19`)

Switches the motor's communication protocol. Takes effect after power cycle.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x19` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte6 (F_CMD) | `0x00` = Private (default); `0x01` = CANopen; `0x02` = MIT |
| Byte0~5 | `01 02 03 04 05 06` (fixed padding) |

**Response:** Type 0 broadcast reply frame.

---

### Type 26 — Version Number Read (`0x04` with special payload)

Reads the firmware version number from the motor.

**Command Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x4` |
| Bit15~8 | Host CAN ID |
| Bit7~0 | Target motor CAN ID |
| Byte0 | `0x00` |
| Byte1 | `0xC4` |
| Byte2~7 | (unused) |

**Reply Frame:**

| Field | Value |
|---|---|
| Bit28~24 | `0x2` |
| Bit15~8 | Motor CAN ID |
| Bit23~16 | Fault flags (same as Type 2) |
| Bit7~0 | Host CAN ID |
| Byte0 | `0x00` |
| Byte1 | `0xC4` |
| Byte2 | `0x56` |
| Byte3~6 | Motor firmware version number (high byte first) |

---

## Read/Write Parameter Index Table

Used with Type 17 (read) and Type 18 (write). All indices are 16-bit, sent little-endian.

### Control / Mode Parameters

| Index | Name | Type | Bytes | Description | RS00 Range | RS04 Range | RS05 Range | R/W |
|---|---|---|---|---|---|---|---|---|
| `0x7005` | `run_mode` | uint8 | 1 | Operating mode: `0`=Operation ctrl, `1`=PP position, `2`=Velocity, `3`=Current, `5`=CSP position | same | same | same | W/R |
| `0x7006` | `iq_ref` | float | 4 | Current mode Iq command | −16 ~ +16 A | −90 ~ +90 A | −11 ~ +11 A | W/R |
| `0x700A` | `spd_ref` | float | 4 | Velocity mode speed command | −33 ~ +33 rad/s | −20 ~ +20 rad/s | −50 ~ +50 rad/s | W/R |
| `0x700B` | `limit_torque` | float | 4 | Torque limit | 0 ~ 14 Nm | 0 ~ 120 Nm | 0 ~ 5.5 Nm | W/R |
| `0x7016` | `loc_ref` | float | 4 | Position mode angle command | rad | rad | rad | W/R |
| `0x7017` | `limit_spd` | float | 4 | CSP position mode speed limit | 0 ~ 33 rad/s | 0 ~ 20 rad/s | 0 ~ 50 rad/s | W/R |
| `0x7018` | `limit_cur` | float | 4 | Velocity/position mode current limit | 0 ~ 16 A | 0 ~ 90 A | 0 ~ 11 A | W/R |

### Current Loop Parameters

| Index | Name | Type | Bytes | Description | Default | R/W |
|---|---|---|---|---|---|---|
| `0x7010` | `cur_kp` | float | 4 | Current loop Kp | 0.17 (all) | W/R |
| `0x7011` | `cur_ki` | float | 4 | Current loop Ki | 0.012 (all) | W/R |
| `0x7014` | `cur_filt_gain` | float | 4 | Current filter gain | 0 ~ 1.0, default 0.1 | W/R |

### Velocity Loop Parameters

| Index | Name | Type | Bytes | Description | RS00 Default | RS04 Default | RS05 Default | R/W |
|---|---|---|---|---|---|---|---|---|
| `0x701F` | `spd_kp` | float | 4 | Velocity loop Kp | 6 | 6 | 6 | W/R |
| `0x7020` | `spd_ki` | float | 4 | Velocity loop Ki | 0.02 | 0.02 | 0.02 | W/R |
| `0x7021` | `spd_filt_gain` | float | 4 | Velocity filter gain | 0.1 | 0.1 | 0.1 | W/R (RS05: W only) |
| `0x7022` | `acc_rad` | float | 4 | Velocity mode acceleration | 20 rad/s² | 15 rad/s² | 20 rad/s² | W/R (RS05: W only) |

### Position Loop Parameters

| Index | Name | Type | Bytes | Description | RS00 Default | RS04 Default | RS05 Default | R/W |
|---|---|---|---|---|---|---|---|---|
| `0x701E` | `loc_kp` | float | 4 | Position loop Kp | 40 | 60 | 40 | W/R |
| `0x7024` | `vel_max` | float | 4 | PP position mode max speed | 10 rad/s | 10 rad/s | 10 rad/s | W/R (RS05: W only) |
| `0x7025` | `acc_set` | float | 4 | PP position mode acceleration | 10 rad/s² | 10 rad/s² | 10 rad/s² | W/R (RS05: W only) |

### Read-Only Feedback Parameters

| Index | Name | Type | Bytes | Description | RS00 Range | RS04 Range | RS05 Range | R/W |
|---|---|---|---|---|---|---|---|---|
| `0x7019` | `mechPos` | float | 4 | Mechanical angle of the load | rad | rad | rad | R |
| `0x701A` | `iqf` | float | 4 | Iq (filtered) | −16 ~ +16 A | −90 ~ +90 A | −11 ~ +11 A | R |
| `0x701B` | `mechVel` | float | 4 | Load shaft velocity | −33 ~ +33 rad/s | −15 ~ +15 rad/s | −50 ~ +50 rad/s | R |
| `0x701C` | `VBUS` | float | 4 | Bus voltage | V | V | V | R |

### Communication & Timing Parameters

| Index | Name | Type | Bytes | Description | Default | R/W |
|---|---|---|---|---|---|---|
| `0x7026` | `EPScan_time` | uint16 | 2 | Active report interval (1=10ms; each +1 adds 5ms) | 1 | W (RS04/RS00: W/R) |
| `0x7028` | `canTimeout` | uint32 | 4 | CAN timeout threshold (20000 = 1s; 0 = disabled) | 0 | W (RS04/RS00: W/R) |

### Zero / Offset Parameters

| Index | Name | Type | Bytes | Description | Default | R/W |
|---|---|---|---|---|---|---|
| `0x7029` | `zero_sta` | uint8 | 1 | Zero flag: `0`=0~2π range; `1`=−π~π range | 0 | W |
| `0x702B` | `add_offset` | float | 4 | Position zero offset (rad) — shifts zero point by this value | 0 | W/R |

### Damping Parameter *(RS00 and RS04 only)*

| Index | Name | Type | Bytes | Description | Default | R/W |
|---|---|---|---|---|---|---|
| `0x702A` | `damper` | uint8 | 1 | Post-power-off anti-backdrive damping switch: `0`=enabled (default); `1`=disabled | 0 | W/R |

---

## CANopen Protocol Switching Frame (Extended Frame)

Sent as a 29-bit extended frame to switch the motor protocol. Takes effect after power cycle.

**Frame:**

| Field | Value |
|---|---|
| Bit28~0 | `0xFFF` |
| Byte0~5 | `01 02 03 04 05 06` (fixed padding) |
| Byte6 (F_CMD) | `0x00` = Private; `0x01` = CANopen; `0x02` = MIT |

**Response:**

| Field | Value |
|---|---|
| Bit10~0 | Motor ID |
| Byte0~7 | 64-bit MCU unique identifier |

---

## MIT Protocol — Standard Frame Messages

The MIT protocol uses an **11-bit standard CAN ID** (not the extended 29-bit frame). Default baud rate: 1 Mbps.

### MIT Frame Structure

| Field | Bits | Description |
|---|---|---|
| Mode type | Bit10~8 | Command/mode identifier |
| Motor ID | Bit7~0 | Target motor CAN ID |
| Data | Byte0~7 | Payload |

---

### MIT Response Command 1 — Data Feedback (Motor Status)

Sent by motor in response to most MIT commands.

| Field | Value |
|---|---|
| Bit10~0 | Host CAN ID |
| Byte0 | Motor CAN ID |
| Byte1~2 | Current angle `[0~65535]` → −12.57 ~ +12.57 rad |
| Byte3 (high 8b) + Byte4[7:4] (low 4b) | Current speed `[0~4096]` → −50 ~ +50 rad/s |
| Byte4[3:0] (high 4b) + Byte5 (low 8b) | Current torque `[0~4096]` → torque range |
| Byte6~7 | Winding temperature: `Temp(°C) × 10` |

---

### MIT Response Command 2 — MCU Identification

| Field | Value |
|---|---|
| Bit10~0 | Motor ID |
| Byte0~7 | 64-bit MCU unique identifier |

---

### MIT Command 1 — Enable Motor Operation

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0~7 | `FF FF FF FF FF FF FF FC` |

Response: MIT Response Command 1.

---

### MIT Command 2 — Stop Motor Operation

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0~7 | `FF FF FF FF FF FF FF FD` |

Response: MIT Response Command 1.

---

### MIT Command 3 — MIT Dynamic Parameters (Motion Control)

Sends all five motion control parameters simultaneously.

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0~1 | Target angle `[0~65535]` → −12.57 ~ +12.57 rad |
| Byte2 (high 8b) + Byte3[7:4] (low 4b) | Target speed `[0~4096]` → −50 ~ +50 rad/s |
| Byte3[3:0] (high 4b) + Byte4 (low 8b) | Kp `[0~4096]` → 0 ~ 500 |
| Byte5 (high 8b) + Byte6[7:4] (low 4b) | Kd `[0~4096]` → 0 ~ 5 |
| Byte6[3:0] (high 4b) + Byte7 (low 8b) | Target torque `[0~4096]` → torque range |

Response: MIT Response Command 1.

---

### MIT Command 4 — Set Zero Position

Sets the current position as zero. Not available in position modes.

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0~7 | `FF FF FF FF FF FF FF FE` |

Response: MIT Response Command 1.

---

### MIT Command 5 — Clear Errors & Read Fault Status

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0~5 | `FF FF FF FF FF FF` |
| Byte6 (F_CMD) | `0xFF` = clear fault; any other value = read fault status |
| Byte7 | `FB` |

**Fault Status Response:**

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0 | Motor CAN ID |
| Byte1~4 | Fault value (non-zero = fault; 0 = normal) |

Fault bits: Bit14=stall/I²t overload, Bit7=encoder uncalibrated, Bit3=overvoltage, Bit2=undervoltage, Bit1=driver IC fault, Bit0=overtemperature (135°C default).

---

### MIT Command 7 — Modify Motor CAN ID

| Field | Value |
|---|---|
| Bit10~0 | Current target motor CAN ID |
| Byte0~5 | `FF FF FF FF FF FF` |
| Byte6 (F_CMD) | New CAN ID |
| Byte7 | `FA` |

Response: MIT Response Command 2.

---

### MIT Command 8 — Change Communication Protocol

Takes effect after power cycle.

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0~5 | `FF FF FF FF FF FF` |
| Byte6 (F_CMD) | `0x00`=Private; `0x01`=CANopen; `0x02`=MIT |
| Byte7 | `FD` |

Response: MIT Response Command 2.

---

### MIT Command 9 — Modify Host CAN ID

| Field | Value |
|---|---|
| Bit10~0 | Target motor CAN ID |
| Byte0~5 | `FF FF FF FF FF FF` |
| Byte6 (F_CMD) | New host CAN ID |
| Byte7 | `01` |

Response: MIT Response Command 2.

---

### MIT Command 10 — Position Mode Control

| Field | Value |
|---|---|
| Bit10~8 | `1` (mode type) |
| Bit7~0 | Target motor CAN ID |
| Byte0~3 | Target position (rad, 32-bit IEEE-754 float) |
| Byte4~7 | Target speed (rad/s, 32-bit IEEE-754 float) |

Response: MIT Response Command 1.

---

### MIT Command 11 — Velocity Mode Control

| Field | Value |
|---|---|
| Bit10~8 | `2` (mode type) |
| Bit7~0 | Target motor CAN ID |
| Byte0~3 | Target speed (rad/s, 32-bit IEEE-754 float) |
| Byte4~7 | Current limit in speed/position mode (A, 32-bit IEEE-754 float) |

Response: MIT Response Command 1.

---

## Motor Specification Summary

| Parameter | RS00 | RS04 | RS05 |
|---|---|---|---|
| Peak torque | 14 Nm | 120 Nm | 5.5 Nm |
| Max speed | 33 rad/s | 15 rad/s | 50 rad/s |
| Max phase current | 16 A | 90 A | 11 A |
| Overtemp fault threshold | 135°C | 145°C | 135°C |
| CAN baud rate | 1 Mbps | 1 Mbps | 1 Mbps |
| CAN frame type (private) | Extended (29-bit) | Extended (29-bit) | Extended (29-bit) |
| Damper parameter (`0x702A`) | ✅ | ✅ | ❌ |

---

## Typical Command Sequences

### Operation Control Mode (MIT-style, Private Protocol)
1. Send **Type 3** (Enable Motor)
2. Send **Type 1** (Motion control command with target pos/vel/torque/Kp/Kd)
3. Receive **Type 2** (Motor feedback)
4. Send **Type 4** (Stop) when done

### Current Mode
1. Send **Type 18** — write `run_mode` (`0x7005`) = `3`
2. Send **Type 3** (Enable)
3. Send **Type 18** — write `iq_ref` (`0x7006`) = desired current (A)
4. Receive **Type 2** feedback

### Velocity Mode
1. Send **Type 18** — write `run_mode` (`0x7005`) = `2`
2. Send **Type 18** — write `limit_cur` (`0x7018`) = max current
3. Send **Type 3** (Enable)
4. Send **Type 18** — write `spd_ref` (`0x700A`) = desired speed (rad/s)
5. Receive **Type 2** feedback

### CSP Position Mode
1. Send **Type 18** — write `run_mode` (`0x7005`) = `5`
2. Send **Type 18** — write `limit_spd` (`0x7017`) = max speed
3. Send **Type 3** (Enable)
4. Send **Type 18** — write `loc_ref` (`0x7016`) = desired position (rad)
5. Receive **Type 2** feedback

### PP Position Mode
1. Send **Type 18** — write `run_mode` (`0x7005`) = `1`
2. Send **Type 18** — write `vel_max` (`0x7024`) = max profile speed
3. Send **Type 18** — write `acc_set` (`0x7025`) = profile acceleration
4. Send **Type 3** (Enable)
5. Send **Type 18** — write `loc_ref` (`0x7016`) = target position (rad)
6. Receive **Type 2** feedback

### Save Parameters to Flash
After any Type 18 write that needs to persist across power cycles:

1. Send **Type 22** (Motor Data Save)

---

*Source: RobStride Motor User Manual — RS00, RS04, RS05 combined edition.*
