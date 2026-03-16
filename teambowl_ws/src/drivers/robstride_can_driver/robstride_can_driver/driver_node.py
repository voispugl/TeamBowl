#!/usr/bin/env python3
"""
RobStride CAN Motor Driver Node — ROS2 Humble

Subscribes to /joint_commands (sensor_msgs/JointState) and publishes
/joint_states at 100 Hz. Uses Operation Control Mode (Type 1 Private Protocol)
over SocketCAN.

Startup modes (set in motors.yaml):
    startup_safe : enable motors → read mechPos → hold current position
    startup_home : enable motors → command home_position_rad per joint
"""

import atexit
import math
import signal
import struct
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import can
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

# Custom services
from robstride_can_driver.srv import (
    SetGains,
    SetMotorId,
    SetZero,
    ShiftZero,
    SetZeroOffset,
    ReadMotorParam,
    WriteMotorParam,
)
from robstride_can_driver.can_protocol import (
    build_type1_id,
    build_type1_data,
    build_type3_id,
    build_type3_data,
    build_type4_id,
    build_type4_data,
    build_type6_id,
    build_type6_data,
    build_type7_id,
    build_type17_id,
    build_type17_data,
    build_type18_id,
    build_type18_data,
    build_type22_id,
    build_type22_data,
    build_type24_id,
    build_type24_data,
    decode_type2_frame,
    decode_type2_active,
    decode_type17_reply,
    scale_to_raw,
    get_comm_type,
    MotorFeedback,
)
from robstride_can_driver.motor_config import load_config, DriverConfig, MotorConfig

# ---------------------------------------------------------------------------
# Module-level parameter index constants
# ---------------------------------------------------------------------------
PARAM_ADD_OFFSET = 0x702B
PARAM_MECH_POS = 0x7019


class RobstrideCanDriverNode(Node):
    """ROS2 Humble node that drives RobStride motors over SocketCAN."""

    def __init__(self):
        super().__init__('robstride_can_driver')

        # --- Config ---
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        if not config_file:
            config_path = Path(__file__).parent.parent / 'config' / 'motors.yaml'
        else:
            config_path = Path(config_file)

        self.cfg: DriverConfig = load_config(config_path)

        # --- Thread-safety ---
        self._state_lock = threading.Lock()

        # --- Per-motor state ---
        self._motor_states: Dict[str, Optional[MotorFeedback]] = {
            name: None for name in self.cfg.motors
        }
        self._param_replies: Dict[str, Optional[object]] = {
            name: None for name in self.cfg.motors
        }
        self._param_reply_events: Dict[str, threading.Event] = {
            name: threading.Event() for name in self.cfg.motors
        }

        # --- Open CAN buses ---
        self._buses: Dict[str, can.Bus] = {}
        for bus_name, bus_cfg in self.cfg.buses.items():
            try:
                bus = can.Bus(
                    interface='socketcan',
                    channel=bus_cfg.interface,
                    bitrate=1000000,
                )
                self._buses[bus_name] = bus
                self.get_logger().info(
                    f"Opened CAN bus '{bus_name}' on interface '{bus_cfg.interface}'"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to open CAN bus '{bus_name}' ({bus_cfg.interface}): {exc}"
                )
                raise

        # --- Shutdown hooks ---
        atexit.register(self._shutdown)
        signal.signal(signal.SIGINT, self._sigint_handler)

        # --- RX threads (one per bus) ---
        self._running = True
        self._rx_threads: Dict[str, threading.Thread] = {}
        for bus_name in self._buses:
            t = threading.Thread(
                target=self._rx_thread,
                args=(bus_name,),
                daemon=True,
                name=f"rx_{bus_name}",
            )
            t.start()
            self._rx_threads[bus_name] = t

        # --- Startup CAN commands ---
        # Enable active reporting on all motors
        for motor in self.cfg.motors.values():
            arb_id = build_type24_id(self.cfg.host_can_id, motor.can_id)
            data = build_type24_data(enable=True)
            self._send(motor.bus, arb_id, data)

        # Enable all motors (Type 3)
        self._enable_all()

        # Startup sequence
        if self.cfg.startup_mode == 'startup_safe':
            self._startup_safe()
        elif self.cfg.startup_mode == 'startup_home':
            self._startup_home()

        # --- Publishers ---
        self._pub_joint_states = self.create_publisher(JointState, '/joint_states', 10)
        self._pub_faults = self.create_publisher(DiagnosticArray, '/motor_faults', 10)

        # --- Subscribers ---
        self.create_subscription(JointState, '/joint_commands', self._on_joint_commands, 10)
        self.create_subscription(Bool, '/e_stop', self._on_e_stop, 10)

        # --- Services ---
        self.create_service(Trigger, '/enable_motors', self._srv_enable_motors)
        self.create_service(Trigger, '/stop_motors', self._srv_stop_motors)
        self.create_service(SetGains, '/set_gains', self._srv_set_gains)
        self.create_service(SetMotorId, '/set_motor_id', self._srv_set_motor_id)
        self.create_service(SetZero, '/set_zero', self._srv_set_zero)
        self.create_service(ShiftZero, '/shift_zero', self._srv_shift_zero)
        self.create_service(SetZeroOffset, '/set_zero_offset', self._srv_set_zero_offset)
        self.create_service(ReadMotorParam, '/read_motor_param', self._srv_read_motor_param)
        self.create_service(WriteMotorParam, '/write_motor_param', self._srv_write_motor_param)
        self.create_service(Trigger, '/save_motor_params', self._srv_save_motor_params)

        # --- Control timer ---
        self._control_timer = self.create_timer(
            1.0 / self.cfg.loop_rate_hz, self._control_loop
        )

        self.get_logger().info(
            f"RobStride CAN driver started. Startup mode: {self.cfg.startup_mode}"
        )

    # -----------------------------------------------------------------------
    # CAN send helper
    # -----------------------------------------------------------------------

    def _send(self, bus_name: str, arb_id: int, data: bytes) -> None:
        """Send a single CAN frame. Thread-safe (python-can send is thread-safe)."""
        try:
            bus = self._buses[bus_name]
            msg = can.Message(
                arbitration_id=arb_id,
                data=data,
                is_extended_id=True,
            )
            bus.send(msg)
        except Exception as exc:
            self.get_logger().error(
                f"CAN send error on '{bus_name}' (arb_id=0x{arb_id:08X}): {exc}"
            )

    # -----------------------------------------------------------------------
    # RX thread
    # -----------------------------------------------------------------------

    def _rx_thread(self, bus_name: str) -> None:
        """Receive CAN frames on one bus and update motor state."""
        bus = self._buses[bus_name]
        while self._running:
            msg = bus.recv(timeout=0.1)
            if msg is None or not msg.is_extended_id:
                continue
            comm_type = get_comm_type(msg.arbitration_id)

            if comm_type in (0x02, 0x18):  # Type 2 feedback or Type 24 active report
                motor_id = (msg.arbitration_id >> 8) & 0xFF
                motor = self._find_motor_by_bus_and_id(bus_name, motor_id)
                if motor is None:
                    continue
                spec = self.cfg.get_spec(motor.motor_type)
                feedback = decode_type2_frame(
                    msg.arbitration_id,
                    msg.data,
                    spec.velocity_min,
                    spec.velocity_max,
                    spec.torque_min,
                    spec.torque_max,
                )
                with self._state_lock:
                    self._motor_states[motor.joint_name] = feedback

            elif comm_type == 0x11:  # Type 17 param read reply
                motor_id = (msg.arbitration_id >> 8) & 0xFF
                motor = self._find_motor_by_bus_and_id(bus_name, motor_id)
                if motor is None:
                    continue
                reply = decode_type17_reply(msg.arbitration_id, msg.data)
                with self._state_lock:
                    self._param_replies[motor.joint_name] = reply
                self._param_reply_events[motor.joint_name].set()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _find_motor_by_bus_and_id(
        self, bus_name: str, motor_id: int
    ) -> Optional[MotorConfig]:
        """Return the MotorConfig matching (bus_name, can_id), or None."""
        for motor in self.cfg.motors_on_bus(bus_name):
            if motor.can_id == motor_id:
                return motor
        return None

    def _get_motor(self, joint_name: str) -> Optional[MotorConfig]:
        """Return the MotorConfig for the given joint name, logging a warning if absent."""
        motor = self.cfg.motors.get(joint_name)
        if motor is None:
            self.get_logger().warn(f"Unknown joint name: '{joint_name}'")
        return motor

    def _read_param_sync(
        self,
        motor: MotorConfig,
        param_index: int,
        timeout: float = 0.5,
    ) -> Optional[bytes]:
        """
        Send a Type 17 read request and block until a reply arrives or timeout.

        Returns the 4 raw value bytes, or None on timeout.
        """
        event = self._param_reply_events[motor.joint_name]
        event.clear()
        with self._state_lock:
            self._param_replies[motor.joint_name] = None

        arb_id = build_type17_id(self.cfg.host_can_id, motor.can_id)
        data = build_type17_data(param_index)
        self._send(motor.bus, arb_id, data)

        if not event.wait(timeout=timeout):
            self.get_logger().warn(
                f"Timeout waiting for param reply from '{motor.joint_name}' "
                f"(index=0x{param_index:04X})"
            )
            return None

        with self._state_lock:
            reply = self._param_replies[motor.joint_name]

        if reply is None or not reply.success:
            return None
        return reply.value_bytes

    # -----------------------------------------------------------------------
    # Startup sequences
    # -----------------------------------------------------------------------

    def _startup_safe(self) -> None:
        """Read mechPos for each motor and hold that position."""
        for motor in self.cfg.motors.values():
            value_bytes = self._read_param_sync(motor, PARAM_MECH_POS, timeout=0.5)
            if value_bytes is not None:
                position = struct.unpack('<f', value_bytes)[0]
                motor.commanded_position = position
                motor.commanded_velocity = 0.0
                self.get_logger().info(
                    f"startup_safe: '{motor.joint_name}' holding at {position:.4f} rad"
                )
            else:
                self.get_logger().warn(
                    f"Could not read mechPos for {motor.joint_name}, defaulting to 0.0"
                )
                motor.commanded_position = 0.0
                motor.commanded_velocity = 0.0

    def _startup_home(self) -> None:
        """Command each motor to its YAML home_position_rad."""
        for motor in self.cfg.motors.values():
            motor.commanded_position = motor.home_position_rad
            motor.commanded_velocity = 0.0
            self.get_logger().info(
                f"startup_home: '{motor.joint_name}' → {motor.home_position_rad:.4f} rad"
            )

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    def _control_loop(self) -> None:
        """100 Hz timer: send Type 1 motion commands and publish joint states / faults."""
        for motor in self.cfg.motors.values():
            if motor.commanded_position is None:
                continue

            spec = self.cfg.get_spec(motor.motor_type)

            torque_raw = scale_to_raw(
                motor.current_torque_ff, spec.torque_min, spec.torque_max
            )
            angle_raw = scale_to_raw(
                motor.commanded_position, spec.position_min, spec.position_max
            )
            vel_raw = scale_to_raw(
                motor.commanded_velocity or 0.0, spec.velocity_min, spec.velocity_max
            )
            kp_raw = scale_to_raw(motor.current_kp, spec.kp_min, spec.kp_max)
            kd_raw = scale_to_raw(motor.current_kd, spec.kd_min, spec.kd_max)

            arb_id = build_type1_id(motor.can_id, torque_raw)
            data = build_type1_data(angle_raw, vel_raw, kp_raw, kd_raw)
            self._send(motor.bus, arb_id, data)

        self._publish_joint_states()
        self._publish_faults()

    # -----------------------------------------------------------------------
    # Publishers
    # -----------------------------------------------------------------------

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        with self._state_lock:
            states_snapshot = dict(self._motor_states)

        for joint_name in self.cfg.motors:
            fb = states_snapshot.get(joint_name)
            msg.name.append(joint_name)
            msg.position.append(fb.position if fb is not None else 0.0)
            msg.velocity.append(fb.velocity if fb is not None else 0.0)
            msg.effort.append(fb.torque if fb is not None else 0.0)

        self._pub_joint_states.publish(msg)

    def _publish_faults(self) -> None:
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        with self._state_lock:
            states_snapshot = dict(self._motor_states)

        for joint_name, fb in states_snapshot.items():
            status = DiagnosticStatus()
            status.name = f"robstride_can_driver/{joint_name}"
            status.hardware_id = joint_name

            if fb is None:
                status.level = DiagnosticStatus.WARN
                status.message = "No feedback received"
            else:
                any_fault = any([
                    fb.fault_uncalibrated,
                    fb.fault_overload,
                    fb.fault_encoder,
                    fb.fault_overtemp,
                    fb.fault_overcurrent,
                    fb.fault_undervoltage,
                ])
                status.level = DiagnosticStatus.ERROR if any_fault else DiagnosticStatus.OK
                status.message = "Fault detected" if any_fault else "OK"
                status.values = [
                    KeyValue(key='temperature_C', value=str(fb.temperature)),
                    KeyValue(key='fault_uncalibrated', value=str(fb.fault_uncalibrated)),
                    KeyValue(key='fault_overload', value=str(fb.fault_overload)),
                    KeyValue(key='fault_encoder', value=str(fb.fault_encoder)),
                    KeyValue(key='fault_overtemp', value=str(fb.fault_overtemp)),
                    KeyValue(key='fault_overcurrent', value=str(fb.fault_overcurrent)),
                    KeyValue(key='fault_undervoltage', value=str(fb.fault_undervoltage)),
                    KeyValue(key='mode', value=str(fb.mode)),
                ]

            diag_array.status.append(status)

        self._pub_faults.publish(diag_array)

    # -----------------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------------

    def _on_joint_commands(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            motor = self._get_motor(name)
            if motor is None:
                continue
            if idx < len(msg.position):
                motor.commanded_position = msg.position[idx]
            if idx < len(msg.velocity):
                motor.commanded_velocity = msg.velocity[idx]
            if idx < len(msg.effort):
                motor.current_torque_ff = msg.effort[idx]

    def _on_e_stop(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().warn("E-stop received — stopping all motors.")
            self._stop_all()

    # -----------------------------------------------------------------------
    # Motor enable / stop helpers
    # -----------------------------------------------------------------------

    def _stop_all(self, clear_fault: bool = False) -> None:
        """Send Type 4 stop command to every motor."""
        for motor in self.cfg.motors.values():
            arb_id = build_type4_id(self.cfg.host_can_id, motor.can_id, clear_fault)
            data = build_type4_data(clear_fault)
            self._send(motor.bus, arb_id, data)

    def _enable_all(self) -> None:
        """Send Type 3 enable command to every motor."""
        for motor in self.cfg.motors.values():
            arb_id = build_type3_id(self.cfg.host_can_id, motor.can_id)
            data = build_type3_data()
            self._send(motor.bus, arb_id, data)

    # -----------------------------------------------------------------------
    # Service callbacks
    # -----------------------------------------------------------------------

    def _srv_enable_motors(self, request, response):
        self._enable_all()
        response.success = True
        response.message = "All motors enabled."
        return response

    def _srv_stop_motors(self, request, response):
        self._stop_all()
        response.success = True
        response.message = "All motors stopped."
        return response

    def _srv_set_gains(self, request, response):
        if request.joint_name == 'all':
            for motor in self.cfg.motors.values():
                motor.current_kp = request.kp
                motor.current_kd = request.kd
        else:
            motor = self._get_motor(request.joint_name)
            if motor is None:
                response.success = False
                response.message = f"Unknown joint: '{request.joint_name}'"
                return response
            motor.current_kp = request.kp
            motor.current_kd = request.kd

        response.success = True
        response.message = f"Gains set: kp={request.kp}, kd={request.kd}"
        return response

    def _srv_set_motor_id(self, request, response):
        motor = self._get_motor(request.joint_name)
        if motor is None:
            response.success = False
            response.message = f"Unknown joint: '{request.joint_name}'"
            return response

        arb_id = build_type7_id(self.cfg.host_can_id, motor.can_id, request.new_can_id)
        # Type 7 uses zero data bytes
        self._send(motor.bus, arb_id, bytes(8))
        motor.can_id = int(request.new_can_id)

        response.success = True
        response.message = (
            f"Motor '{request.joint_name}' CAN ID changed to {request.new_can_id}"
        )
        return response

    def _srv_set_zero(self, request, response):
        if request.joint_name == 'all':
            for motor in self.cfg.motors.values():
                arb_id = build_type6_id(self.cfg.host_can_id, motor.can_id)
                self._send(motor.bus, arb_id, build_type6_data())
        else:
            motor = self._get_motor(request.joint_name)
            if motor is None:
                response.success = False
                response.message = f"Unknown joint: '{request.joint_name}'"
                return response
            arb_id = build_type6_id(self.cfg.host_can_id, motor.can_id)
            self._send(motor.bus, arb_id, build_type6_data())

        response.success = True
        response.message = f"Set zero sent to '{request.joint_name}'"
        return response

    def _srv_shift_zero(self, request, response):
        if request.joint_name == 'all':
            motors_to_update = list(self.cfg.motors.values())
        else:
            motor = self._get_motor(request.joint_name)
            if motor is None:
                response.success = False
                response.message = f"Unknown joint: '{request.joint_name}'"
                response.new_offset = 0.0
                return response
            motors_to_update = [motor]

        last_new_offset = 0.0
        for motor in motors_to_update:
            # Read current add_offset
            value_bytes = self._read_param_sync(motor, PARAM_ADD_OFFSET, timeout=0.5)
            if value_bytes is not None:
                current_offset = struct.unpack('<f', value_bytes)[0]
            else:
                self.get_logger().warn(
                    f"shift_zero: could not read add_offset for '{motor.joint_name}', "
                    f"assuming 0.0"
                )
                current_offset = 0.0

            new_offset = current_offset + request.delta_rad
            last_new_offset = new_offset

            arb_id = build_type18_id(self.cfg.host_can_id, motor.can_id)
            data = build_type18_data(PARAM_ADD_OFFSET, new_offset, value_type='float')
            self._send(motor.bus, arb_id, data)

        response.success = True
        response.message = f"shift_zero applied to '{request.joint_name}'"
        response.new_offset = last_new_offset
        return response

    def _srv_set_zero_offset(self, request, response):
        if request.joint_name == 'all':
            motors_to_update = list(self.cfg.motors.values())
        else:
            motor = self._get_motor(request.joint_name)
            if motor is None:
                response.success = False
                response.message = f"Unknown joint: '{request.joint_name}'"
                return response
            motors_to_update = [motor]

        for motor in motors_to_update:
            arb_id = build_type18_id(self.cfg.host_can_id, motor.can_id)
            data = build_type18_data(PARAM_ADD_OFFSET, request.offset_rad, value_type='float')
            self._send(motor.bus, arb_id, data)

        response.success = True
        response.message = (
            f"add_offset set to {request.offset_rad} rad on '{request.joint_name}'"
        )
        return response

    def _srv_read_motor_param(self, request, response):
        motor = self._get_motor(request.joint_name)
        if motor is None:
            response.success = False
            response.message = f"Unknown joint: '{request.joint_name}'"
            response.value_float = 0.0
            response.value_uint32 = 0
            return response

        value_bytes = self._read_param_sync(motor, request.param_index, timeout=0.5)
        if value_bytes is None:
            response.success = False
            response.message = (
                f"Timeout or error reading param 0x{request.param_index:04X} "
                f"from '{request.joint_name}'"
            )
            response.value_float = 0.0
            response.value_uint32 = 0
            return response

        value_float = struct.unpack('<f', value_bytes)[0]
        value_uint32 = struct.unpack('<I', value_bytes)[0]

        response.success = True
        response.message = (
            f"param 0x{request.param_index:04X}: float={value_float}, uint32={value_uint32}"
        )
        response.value_float = float(value_float)
        response.value_uint32 = int(value_uint32)
        return response

    def _srv_write_motor_param(self, request, response):
        motor = self._get_motor(request.joint_name)
        if motor is None:
            response.success = False
            response.message = f"Unknown joint: '{request.joint_name}'"
            return response

        valid_types = ('float', 'uint8', 'uint16', 'uint32')
        if request.value_type not in valid_types:
            response.success = False
            response.message = (
                f"Invalid value_type '{request.value_type}'. "
                f"Must be one of: {valid_types}"
            )
            return response

        arb_id = build_type18_id(self.cfg.host_can_id, motor.can_id)
        data = build_type18_data(
            request.param_index, request.value, value_type=request.value_type
        )
        self._send(motor.bus, arb_id, data)

        response.success = True
        response.message = (
            f"Wrote {request.value} ({request.value_type}) to "
            f"param 0x{request.param_index:04X} on '{request.joint_name}'"
        )
        return response

    def _srv_save_motor_params(self, request, response):
        for motor in self.cfg.motors.values():
            arb_id = build_type22_id(self.cfg.host_can_id, motor.can_id)
            data = build_type22_data()
            self._send(motor.bus, arb_id, data)

        response.success = True
        response.message = "Save (Type 22) sent to all motors."
        return response

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Stop all motors, cancel timer, join RX threads, close buses."""
        if not self._running:
            return
        self._running = False

        try:
            self._control_timer.cancel()
        except Exception:
            pass

        try:
            self._stop_all()
        except Exception:
            pass

        for name, t in self._rx_threads.items():
            t.join(timeout=1.0)

        for bus_name, bus in self._buses.items():
            try:
                bus.shutdown()
                self.get_logger().info(f"CAN bus '{bus_name}' closed.")
            except Exception as exc:
                self.get_logger().error(f"Error closing bus '{bus_name}': {exc}")

    def _sigint_handler(self, signum, frame) -> None:
        self._shutdown()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RobstrideCanDriverNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node._shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
