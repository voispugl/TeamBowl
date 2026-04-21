#!/usr/bin/env python3
"""
steamdeck_ws_teleop — Steam Deck browser gamepad → Nav2 goal sender + diagnostics + nav map.

Runs a WebSocket server on the robot. The Steam Deck opens a browser to
http://ROBOT_IP:8888 to get the control page, which reads the Steam Deck
gamepad via the Web Gamepad API and streams state over WebSocket.

Two WS message types from server → browser:
  type=status  sent in reply to every joy message (20 Hz)
  type=push    sent on a timer (default 2 Hz): diagnostics + coarse nav map

Controls:
  Hold RT (right trigger)   arm goal accumulation
  Left stick                position goal (forward/back/strafe)
  Right stick X             rotate goal heading
  Release RT                reset goal accumulator to origin
  A button                  send goal to Nav2
  B button                  cancel active navigation / reset goal
  Menu button               E-stop
"""

from __future__ import annotations

import asyncio
import http
import json
import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, String

import websockets
from websockets.asyncio.server import serve as ws_serve
from websockets.datastructures import Headers as WsHeaders
from websockets.http11 import Response as WsResponse

# ---------------------------------------------------------------------------
# Embedded control pages — served at http://ROBOT_IP:8888/
# ui_mode='phone' → _HTML_PHONE (3 big buttons + diagnostics, phone-first)
# ui_mode='full'  → _HTML_FULL  (full gamepad + trajectory + gains + map)
# ---------------------------------------------------------------------------

_HTML_PHONE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TeamBowl</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#111;color:#ddd;padding:16px;max-width:480px;margin:auto}
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}
.hdr h2{color:#4af;font-size:20px}
.ws-dot{width:12px;height:12px;border-radius:50%;background:#f44;display:inline-block;margin-right:6px}
.ws-dot.ok{background:#4f4}
#ws-label{font-size:13px}
.btn{display:block;width:100%;min-height:18vh;font-size:clamp(2.5rem,10vw,5rem);font-weight:bold;
  border:none;border-radius:16px;margin-bottom:16px;cursor:pointer;color:#fff;letter-spacing:2px}
.btn-enable{background:#1a6b1a}
.btn-enable:active{background:#2a9b2a}
.btn-lid{background:#155a8a}
.btn-lid:active{background:#1e7fc0}
.btn-reset{background:#5a3a00}
.btn-reset:active{background:#9a6000}
.btn-kill{background:#7a1a1a}
.btn-kill:active{background:#c02020}
.btn-auton{background:#4a1a6b}
.btn-auton:active{background:#7a2aa0}
.diag{background:#1e1e1e;border:1px solid #333;border-radius:8px;padding:12px}
.diag h3{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #2a2a2a;font-size:15px}
.row:last-child{border-bottom:none}
.key{color:#888}
.val{font-weight:bold}
.ok{color:#4f4}.warn{color:#fa0}.err{color:#f44}.info{color:#4af}.dim{color:#666}
</style>
</head>
<body>
<div class="hdr">
  <h2>TeamBowl</h2>
  <span><span class="ws-dot" id="ws-dot"></span><span id="ws-label">Connecting…</span></span>
</div>

<button class="btn btn-enable" onclick="send({type:'clear_estop'})">ENABLE</button>
<button class="btn btn-auton"  onclick="send({type:'set_mode',mode:'auton'})">AUTON</button>
<button class="btn btn-lid"    onclick="send({type:'lid_cmd',cmd:'toggle'})">TOGGLE LID</button>
<button class="btn btn-reset"  onclick="send({type:'reset_odom'})">RESET ODOM</button>
<button class="btn btn-kill"   onclick="send({type:'estop'})">KILL</button>

<div id="pitch-warn" style="display:none;background:#7a1a00;color:#fff;border-radius:10px;padding:12px 16px;margin-bottom:14px;font-size:clamp(1.4rem,6vw,2.2rem);font-weight:bold;text-align:center;letter-spacing:1px"></div>

<div class="diag">
  <h3>Diagnostics</h3>
  <div class="row"><span class="key">Mode</span>      <span class="val info" id="mode-val">—</span></div>
  <div class="row"><span class="key">E-stop</span>    <span class="val ok"   id="estop-val">—</span></div>
  <div class="row"><span class="key">Stuck</span>     <span class="val ok"   id="stuck-val">—</span></div>
  <div class="row"><span class="key">Kill sw</span>   <span class="val ok"   id="kill-val">—</span></div>
  <div class="row"><span class="key">Lid</span>       <span class="val dim"  id="lid-val">—</span></div>
  <div class="row"><span class="key">Battery</span>   <span class="val dim"  id="battery-val">—</span></div>
  <div class="row"><span class="key">Legs</span>      <span class="val dim"  id="legs-val">—</span></div>
  <div class="row"><span class="key">Planner</span>   <span class="val dim"  id="planner-val">—</span></div>
  <div class="row"><span class="key">Person</span>    <span class="val dim"  id="person-val">—</span></div>
  <div class="row"><span class="key">Pitch</span>      <span class="val dim"  id="pitch-val">—</span></div>
  <div class="row"><span class="key">Ctrl in v/ω</span><span class="val dim" id="ctrl-in-val">—</span></div>
  <div class="row"><span class="key">Motor v/ω</span>  <span class="val dim" id="motor-val">—</span></div>
</div>

<style>
.gpanel{background:#1e1e1e;border:1px solid #333;border-radius:8px;padding:12px;margin-top:12px}
.gpanel h3{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.grow{display:flex;align-items:center;gap:6px;padding:3px 0}
.grow label{flex:0 0 150px;color:#888;font-size:12px;text-align:right}
.grow input{width:80px;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:3px 6px;font-family:monospace;font-size:12px}
.gbtn-row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
.gbtn{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:5px 14px;cursor:pointer;font-family:monospace;font-size:13px}
.gbtn:hover{background:#3a3a3a}
.ginfo{font-size:11px;color:#666;margin-top:4px}
.dtbl{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:6px}
.dtbl th{color:#666;font-weight:normal;text-align:left;padding:2px 4px;border-bottom:1px solid #333}
.dtbl td{padding:2px 4px}
.dtbl input{width:64px;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px;padding:2px 4px;font-family:monospace;font-size:11px}
</style>

<!-- Driving Gains -->
<div class="gpanel" id="p-driving">
  <h3>Driving Gains &nbsp;<span style="font-size:11px;color:#666">θ=<span id="pd-theta">—</span>° v=<span id="pd-v">—</span>m/s ω=<span id="pd-yaw">—</span>rad/s</span></h3>
  <div class="grow"><label>kp_vel</label><input id="pd-kp_vel" type="number" step="0.5"></div>
  <div class="grow"><label>ki_vel</label><input id="pd-ki_vel" type="number" step="0.1"></div>
  <div class="grow"><label>kd_vel</label><input id="pd-kd_vel" type="number" step="0.01"></div>
  <div class="grow"><label>kp_pitch</label><input id="pd-kp_pitch" type="number" step="1"></div>
  <div class="grow"><label>kd_pitch</label><input id="pd-kd_pitch" type="number" step="0.5"></div>
  <div class="grow"><label>ki_pitch</label><input id="pd-ki_pitch" type="number" step="0.1"></div>
  <div class="grow"><label>kp_yaw</label><input id="pd-kp_yaw" type="number" step="0.5"></div>
  <div class="grow"><label>ki_yaw</label><input id="pd-ki_yaw" type="number" step="0.1"></div>
  <div class="grow"><label>kd_yaw</label><input id="pd-kd_yaw" type="number" step="0.05"></div>
  <div class="grow"><label>kff_decel</label><input id="pd-kff_decel" type="number" step="0.01"></div>
  <div class="grow"><label>theta_eq_offset</label><input id="pd-theta_eq_offset" type="number" step="0.005"></div>
  <div class="gbtn-row">
    <button class="gbtn" onclick="drvgReceive()">Receive</button>
    <button class="gbtn" onclick="drvgSend()">Send</button>
    <span id="pd-msg" class="ginfo"></span>
  </div>
</div>

<script>
const wsUrl = 'ws://' + location.host + '/ws';
let ws = null;
let lastDrvGains = {};
const DRV_KEYS = ['kp_vel','ki_vel','kd_vel','kp_pitch','kd_pitch','ki_pitch','kp_yaw','ki_yaw','kd_yaw','kff_decel','theta_eq_offset'];

function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen  = () => setWs(true);
  ws.onclose = () => { setWs(false); setTimeout(connect, 2000); };
  ws.onerror = () => {};
  ws.onmessage = (e) => { try { const d=JSON.parse(e.data); if(d.type==='push') handlePush(d); } catch(_){} };
}
function setWs(ok) {
  document.getElementById('ws-dot').className = 'ws-dot'+(ok?' ok':'');
  document.getElementById('ws-label').textContent = ok ? 'Connected' : 'Disconnected — retrying…';
}
function send(obj) { if(ws && ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify(obj)); }
function set(id, text, cls) {
  const el=document.getElementById(id); if(!el) return;
  el.textContent=text; el.className='val '+(cls||'dim');
}
function setBool(id, val, alarmOnTrue) {
  set(id, val?'YES':'NO', val?(alarmOnTrue?'err':'ok'):(alarmOnTrue?'ok':'warn'));
}
function flash(id, txt) {
  const el=document.getElementById(id); if(!el) return;
  el.textContent=txt; setTimeout(()=>{el.textContent='';},3000);
}

function handlePush(d) {
  set('mode-val', d.mode||'—', d.mode==='driving'?'info':d.mode==='balance'?'warn':'dim');
  setBool('estop-val', d.estop,       true);
  setBool('stuck-val', d.stuck,       true);
  setBool('kill-val',  d.kill_switch, true);
  set('lid-val', d.lid||'—', 'dim');
  if(d.battery_v!=null){const v=d.battery_v;set('battery-val',v.toFixed(1)+' V',v<42?'err':v<44?'warn':'ok');}
  if(d.legs_running!=null)  setBool('legs-val',    d.legs_running,  false);
  if(d.planner_ready!=null) setBool('planner-val', d.planner_ready, false);
  if(d.user_valid!=null)    setBool('person-val',  d.user_valid,    false);
  if(d.ctrl_in_vx!=null) set('ctrl-in-val', d.ctrl_in_vx.toFixed(3)+' / '+d.ctrl_in_wz.toFixed(3), 'dim');
  if(d.motor_vx!=null)   set('motor-val',   d.motor_vx.toFixed(3)+' / '+d.motor_wz.toFixed(3), 'info');
  if(d.driving_gains){try{
    const dg=JSON.parse(d.driving_gains); lastDrvGains=dg;
    if(dg._theta_deg!=null){
      const td=Math.abs(dg._theta_deg);
      set('pitch-val',dg._theta_deg+'°',td>15?'err':td>8?'warn':'ok');
      const pw=document.getElementById('pitch-warn');
      if(pw){
        if(td>15){
          pw.style.display='';
          pw.style.background=td>20?'#9a0000':'#7a3a00';
          pw.textContent='⚠ PITCH '+dg._theta_deg+'° — NEAR FALLOVER';
        } else { pw.style.display='none'; }
      }
    }
    const pt=document.getElementById('pd-theta'); if(pt&&dg._theta_deg!=null) pt.textContent=dg._theta_deg;
    const pv=document.getElementById('pd-v');     if(pv&&dg._v_actual!=null)  pv.textContent=dg._v_actual;
    const pw2=document.getElementById('pd-yaw');   if(pw2&&dg._yaw_dot!=null)   pw2.textContent=dg._yaw_dot;
  }catch(_){}}
}

// Driving gains
function drvgReceive() {
  DRV_KEYS.forEach(k=>{const el=document.getElementById('pd-'+k);if(el&&lastDrvGains[k]!=null) el.value=parseFloat(lastDrvGains[k].toPrecision(6));});
  flash('pd-msg','Received \u2713');
}
function drvgSend() {
  const g={};
  DRV_KEYS.forEach(k=>{const el=document.getElementById('pd-'+k);if(el&&el.value!=='') g[k]=parseFloat(el.value);});
  send({type:'driving_gains',gains:g}); flash('pd-msg','Sent \u2713 '+new Date().toLocaleTimeString());
}

connect();
</script>
</body>
</html>
"""

_HTML_PAGE = _HTML_FULL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TeamBowl Teleop</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#111;color:#ddd;padding:12px;font-size:14px}
h2{color:#4af;margin-bottom:10px;font-size:18px}
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.ws-dot{width:10px;height:10px;border-radius:50%;background:#f44;display:inline-block;margin-right:6px}
.ws-dot.ok{background:#4f4}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.box{background:#1e1e1e;border:1px solid #333;border-radius:6px;padding:10px;margin-bottom:10px}
.box h3{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid #2a2a2a}
.row:last-child{border-bottom:none}
.key{color:#888;font-size:12px}
.val{font-weight:bold;font-size:13px}
.ok{color:#4f4}.warn{color:#fa0}.err{color:#f44}.info{color:#4af}.dim{color:#666}
#map-canvas{display:block;border:1px solid #444;image-rendering:pixelated;width:240px;height:240px}
.map-wrap{display:flex;flex-direction:column;align-items:center;gap:6px}
.map-legend{display:flex;gap:10px;font-size:11px;color:#888}
.swatch{width:12px;height:12px;display:inline-block;vertical-align:middle;margin-right:3px}
.footer{background:#1e1e1e;border:1px solid #333;border-radius:6px;padding:8px 10px;font-size:12px;color:#888;margin-top:4px}
button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:6px 14px;cursor:pointer;font-family:monospace;font-size:13px}
button:hover{background:#3a3a3a;border-color:#666}
button.danger{border-color:#844}
button.danger:hover{background:#3a2020;border-color:#f44}
input[type=number]{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:4px 6px;font-family:monospace;font-size:12px}
input[type=checkbox]{accent-color:#4af;vertical-align:middle}
label{color:#888;font-size:12px}
.btn-row{display:flex;gap:8px;flex-wrap:wrap}
.gains-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:5px;margin-bottom:8px}
.gain-row{display:flex;align-items:center;gap:6px}
.gain-row label{flex:0 0 140px;color:#888;font-size:12px;text-align:right}
.gain-row input{width:78px}
</style>
</head>
<body>

<div class="hdr">
  <h2>TeamBowl Control Panel</h2>
  <span><span class="ws-dot" id="ws-dot"></span><span id="ws-label">Connecting…</span></span>
</div>

<div id="pitch-warn" style="display:none;background:#7a1a00;color:#fff;border-radius:6px;padding:8px 14px;margin-bottom:10px;font-size:16px;font-weight:bold;text-align:center"></div>

<div class="grid">
  <!-- Diagnostics -->
  <div class="box">
    <h3>Diagnostics</h3>
    <div class="row"><span class="key">Gamepad</span><span class="val dim" id="gp-val">—</span></div>
    <div class="row"><span class="key">Dead-man RT</span><span class="val warn" id="dm-val">DISARMED</span></div>
    <div class="row"><span class="key">Mode</span><span class="val info" id="mode-val">—</span></div>
    <div class="row"><span class="key">Nav state</span><span class="val dim" id="nav-val">—</span></div>
    <div class="row"><span class="key">Gamepad goal</span><span class="val dim" id="goal-val">—</span></div>
    <div class="row"><span class="key">Battery</span><span class="val dim" id="battery-val">—</span></div>
    <div class="row"><span class="key">E-stop</span><span class="val ok" id="estop-val">NO</span></div>
    <div class="row"><span class="key">Stuck</span><span class="val ok" id="stuck-val">NO</span></div>
    <div class="row"><span class="key">Kill switch</span><span class="val ok" id="kill-val">NO</span></div>
    <div class="row"><span class="key">Lid</span><span class="val dim" id="lid-val">—</span></div>
    <div class="row"><span class="key">Odom XY</span><span class="val dim" id="odom-xy">—</span></div>
    <div class="row"><span class="key">Odom θ</span><span class="val dim" id="odom-th">—</span></div>
    <div class="row"><span class="key">Planner</span><span class="val dim" id="planner-val">—</span></div>
    <div class="row"><span class="key">Legs</span><span class="val dim" id="legs-val">—</span></div>
    <div class="row"><span class="key">Person</span><span class="val dim" id="person-val">—</span></div>
    <div class="row"><span class="key">Pitch</span><span class="val dim" id="pitch-val">—</span></div>
    <div class="row"><span class="key">Ctrl in (v/ω)</span><span class="val dim" id="ctrl-in-val">—</span></div>
    <div class="row"><span class="key">Motor cmd (v/ω)</span><span class="val dim" id="motor-val">—</span></div>
  </div>

  <!-- Nav map -->
  <div class="box">
    <h3>Nav Map (coarse)</h3>
    <div class="map-wrap">
      <canvas id="map-canvas" width="240" height="240"></canvas>
      <div class="map-legend">
        <span><span class="swatch" style="background:#e8e8e8"></span>free</span>
        <span><span class="swatch" style="background:#555"></span>unknown</span>
        <span><span class="swatch" style="background:#222;border:1px solid #555"></span>occ</span>
        <span><span class="swatch" style="background:#e44;border-radius:50%"></span>robot</span>
        <span><span class="swatch" style="background:#4af"></span>goal</span>
      </div>
      <div id="map-age" style="font-size:11px;color:#666">No map yet</div>
    </div>
  </div>
</div>

<!-- Mode + Lid -->
<div class="grid">
  <div class="box">
    <h3>Robot Mode</h3>
    <div class="btn-row">
      <button onclick="send({type:'set_mode',mode:'driving'})">Driving</button>
      <button onclick="send({type:'set_mode',mode:'balance'})">Balance</button>
      <button onclick="send({type:'set_mode',mode:'auton'})">Auton</button>
      <button class="danger" onclick="send({type:'set_mode',mode:'off'})">Off</button>
      <button style="background:#5a3a00;color:#fff;border-color:#9a6000;font-size:15px;font-weight:bold" onclick="send({type:'reset_odom'})">⟳ Reset Odom</button>
    </div>
  </div>
  <div class="box">
    <h3>Lid &nbsp;<span class="val dim" id="lid-val2">—</span></h3>
    <div class="btn-row">
      <button onclick="send({type:'lid_cmd',cmd:'open'})">Open</button>
      <button onclick="send({type:'lid_cmd',cmd:'close'})">Close</button>
      <button onclick="send({type:'lid_cmd',cmd:'toggle'})">Toggle</button>
    </div>
  </div>
</div>

<!-- Trajectory Goal -->
<div class="box">
  <h3>Trajectory Goal &nbsp;<span class="val dim" id="traj-state">—</span></h3>
  <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
    <label>X&nbsp;<input id="traj-x" type="number" value="2.0" step="0.25" style="width:72px">&nbsp;m</label>
    <label>Y&nbsp;<input id="traj-y" type="number" value="0.0" step="0.25" style="width:72px">&nbsp;m</label>
    <label>θ&nbsp;<input id="traj-th" type="number" value="0.0" step="5" style="width:72px">&nbsp;°</label>
    <label><input id="traj-rel" type="checkbox" checked>&nbsp;Relative</label>
  </div>
  <div class="btn-row">
    <button onclick="trajGo()">&#9654; Go</button>
    <button onclick="send({type:'traj_cmd',cmd:'stop'})">&#9632; Stop</button>
    <button onclick="send({type:'traj_cmd',cmd:'reset'})">&#8635; Reset</button>
  </div>
</div>

<!-- Driving Gains -->
<div class="box">
  <h3>Driving Gains &nbsp;
    <span style="font-size:11px;color:#666">&#952;=<span id="dg-theta">—</span>&#176; &nbsp; v=<span id="dg-v">—</span>&nbsp;m/s &nbsp; &#969;=<span id="dg-yaw">—</span>&nbsp;rad/s</span>
  </h3>
  <div class="gains-grid">
    <div class="gain-row"><label>kp_vel</label><input id="g-kp_vel" type="number" step="0.5"></div>
    <div class="gain-row"><label>ki_vel</label><input id="g-ki_vel" type="number" step="0.1"></div>
    <div class="gain-row"><label>kd_vel</label><input id="g-kd_vel" type="number" step="0.01"></div>
    <div class="gain-row"><label>kp_pitch</label><input id="g-kp_pitch" type="number" step="1"></div>
    <div class="gain-row"><label>kd_pitch</label><input id="g-kd_pitch" type="number" step="0.5"></div>
    <div class="gain-row"><label>ki_pitch</label><input id="g-ki_pitch" type="number" step="0.1"></div>
    <div class="gain-row"><label>kp_yaw</label><input id="g-kp_yaw" type="number" step="0.5"></div>
    <div class="gain-row"><label>ki_yaw</label><input id="g-ki_yaw" type="number" step="0.1"></div>
    <div class="gain-row"><label>kd_yaw</label><input id="g-kd_yaw" type="number" step="0.05"></div>
    <div class="gain-row"><label>kff_decel</label><input id="g-kff_decel" type="number" step="0.01"></div>
    <div class="gain-row"><label>theta_eq_offset</label><input id="g-theta_eq_offset" type="number" step="0.005"></div>
  </div>
  <div class="btn-row">
    <button onclick="drvgReceive()">Receive</button>
    <button onclick="drvgSend()">Send</button>
    <span id="drv-gains-msg" style="font-size:12px;color:#888"></span>
  </div>
</div>

<div class="footer">
  RT=arm &nbsp;|&nbsp; L-stick=position &nbsp;|&nbsp; R-stick X=heading &nbsp;|&nbsp;
  A=send goal &nbsp;|&nbsp; B=cancel &nbsp;|&nbsp; Menu=E-STOP
</div>

<script>
const wsUrl = 'ws://' + location.host + '/ws';
const N = 20;
let ws = null, gpIndex = null, sendInterval = null, prevDm = false;
const GAIN_KEYS = ['kp_vel','ki_vel','kd_vel','kp_pitch','kd_pitch','ki_pitch','kp_yaw','ki_yaw','kd_yaw','kff_decel','theta_eq_offset'];
let lastDrvGainsF = {};

// ---- WebSocket ----
function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen  = () => { setWs(true); startSending(); };
  ws.onclose = () => { setWs(false); stopSending(); setTimeout(connect, 2000); };
  ws.onerror = () => {};
  ws.onmessage = (e) => {
    try { const d = JSON.parse(e.data);
      if (d.type === 'push')   handlePush(d);
      if (d.type === 'status') handleStatus(d);
    } catch(_) {}
  };
}
function setWs(ok) {
  document.getElementById('ws-dot').className    = 'ws-dot' + (ok ? ' ok' : '');
  document.getElementById('ws-label').textContent = ok ? 'Connected' : 'Disconnected — retrying…';
}
function send(obj) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }

// ---- Gamepad ----
window.addEventListener('gamepadconnected',    (e) => { gpIndex = e.gamepad.index; set('gp-val', e.gamepad.id.slice(0,30), 'ok'); });
window.addEventListener('gamepaddisconnected', ()  => { gpIndex = null; set('gp-val', 'Disconnected', 'err'); });
function startSending() { if (!sendInterval) sendInterval = setInterval(sendState, 50); }
function stopSending()  { if (sendInterval) { clearInterval(sendInterval); sendInterval = null; } }
function sendState() {
  if (gpIndex === null || !ws || ws.readyState !== WebSocket.OPEN) return;
  const gp = navigator.getGamepads()[gpIndex];
  if (!gp) return;
  const axes    = Array.from(gp.axes);
  const buttons = gp.buttons.map(b => b.pressed ? 1 : 0);
  const dm = (axes.length > 5 ? axes[5] : 0) > 0.5;
  if (dm !== prevDm) { set('dm-val', dm ? 'ARMED' : 'DISARMED', dm ? 'ok' : 'warn'); prevDm = dm; }
  ws.send(JSON.stringify({axes, buttons}));
}

// ---- Message handlers ----
function handleStatus(d) {
  set('nav-val',  d.state, d.state === 'RUNNING' ? 'info' : 'dim');
  set('goal-val', 'x='+d.goal_x+' y='+d.goal_y+' \u03b8='+d.goal_theta_deg+'\u00b0', 'dim');
}
function handlePush(d) {
  set('mode-val', d.mode||'—', d.mode==='driving'?'info':d.mode==='balance'?'warn':'dim');
  set('nav-val',  d.state||'—', d.state==='RUNNING'?'info':'dim');
  setBool('estop-val', d.estop,       true);
  setBool('stuck-val', d.stuck,       true);
  setBool('kill-val',  d.kill_switch, true);
  set('lid-val',  d.lid||'—', 'dim');
  set('lid-val2', d.lid||'—', 'dim');
  if (d.odom_x != null) {
    set('odom-xy', '('+d.odom_x+', '+d.odom_y+')', 'dim');
    set('odom-th', d.odom_theta_deg+'\u00b0', 'dim');
  }
  if (d.map) drawMap(d);
  document.getElementById('map-age').textContent = 'Updated '+new Date().toLocaleTimeString();

  if (d.battery_v != null) {
    const v = d.battery_v;
    set('battery-val', v.toFixed(1)+' V', v < 42 ? 'err' : v < 44 ? 'warn' : 'ok');
  }
  if (d.planner_ready != null) setBool('planner-val', d.planner_ready, false);
  if (d.legs_running != null)  setBool('legs-val',   d.legs_running,  false);
  if (d.user_valid != null)    setBool('person-val', d.user_valid,    false);
  if (d.ctrl_in_vx  != null) set('ctrl-in-val', d.ctrl_in_vx.toFixed(3)+' / '+d.ctrl_in_wz.toFixed(3), 'dim');
  if (d.motor_vx    != null) set('motor-val',   d.motor_vx.toFixed(3)+' / '+d.motor_wz.toFixed(3), 'info');
  if (d.traj_status) {
    try { const ts = JSON.parse(d.traj_status);
      set('traj-state', ts.state||'—', ts.state==='RUNNING'?'info':'dim');
    } catch(_) {}
  }
  if (d.driving_gains) {
    try { const dg = JSON.parse(d.driving_gains); lastDrvGainsF = dg;
      if (dg._theta_deg != null) {
        const td = Math.abs(dg._theta_deg);
        set('pitch-val', dg._theta_deg+'°', td>15?'err':td>8?'warn':'ok');
        const pw = document.getElementById('pitch-warn');
        if (pw) {
          if (td > 15) {
            pw.style.display = '';
            pw.style.background = td > 20 ? '#9a0000' : '#7a3a00';
            pw.textContent = '⚠ PITCH ' + dg._theta_deg + '° — NEAR FALLOVER';
          } else { pw.style.display = 'none'; }
        }
      }
      const td2 = document.getElementById('dg-theta'); if (td2 && dg._theta_deg != null) td2.textContent = dg._theta_deg;
      const vd = document.getElementById('dg-v');      if (vd  && dg._v_actual  != null) vd.textContent  = dg._v_actual;
      const wd = document.getElementById('dg-yaw');    if (wd  && dg._yaw_dot   != null) wd.textContent  = dg._yaw_dot;
    } catch(_) {}
  }
}


// ---- Trajectory ----
function trajGo() {
  const _pf  = (id) => { const v = parseFloat(document.getElementById(id).value); return isNaN(v) ? 0 : v; };
  const x     = _pf('traj-x');
  const y     = _pf('traj-y');
  const theta = _pf('traj-th') * Math.PI / 180;
  const rel   = document.getElementById('traj-rel').checked;
  send({type:'traj_goal', x, y, theta, relative:rel});
  setTimeout(()=>send({type:'traj_cmd',cmd:'go'}), 80);
}

// ---- Driving Gains ----
function drvgReceive() {
  for (const k of GAIN_KEYS) { const el=document.getElementById('g-'+k); if(el&&lastDrvGainsF[k]!=null) el.value=parseFloat(lastDrvGainsF[k].toPrecision(6)); }
  flash('drv-gains-msg','Received \u2713');
}
function drvgSend() {
  const gains={};
  for (const k of GAIN_KEYS) { const el=document.getElementById('g-'+k); if(el&&el.value!=='') gains[k]=parseFloat(el.value); }
  send({type:'driving_gains', gains});
  flash('drv-gains-msg','Sent \u2713 '+new Date().toLocaleTimeString());
}

function flash(id, txt) { const m=document.getElementById(id); if(!m) return; m.textContent=txt; setTimeout(()=>{m.textContent='';},3000); }

// ---- Helpers ----
function set(id, text, cls) {
  const el = document.getElementById(id); if (!el) return;
  el.textContent = text; el.className = 'val '+(cls||'dim');
}
function setBool(id, val, alarmOnTrue) {
  set(id, val?'YES':'NO', val?(alarmOnTrue?'err':'ok'):(alarmOnTrue?'ok':'warn'));
}

// ---- Map ----
function drawMap(d) {
  const canvas = document.getElementById('map-canvas');
  const ctx = canvas.getContext('2d');
  const sz  = canvas.width / N;
  ctx.fillStyle='#111'; ctx.fillRect(0,0,canvas.width,canvas.height);
  for (let row=0;row<N;row++) for (let col=0;col<N;col++) {
    const c = d.map[row*N+col];
    ctx.fillStyle = c==='0'?'#e8e8e8':c==='1'?'#1a1a1a':'#505050';
    ctx.fillRect(col*sz,(N-1-row)*sz,sz,sz);
  }
  ctx.strokeStyle='#2a2a2a'; ctx.lineWidth=0.5;
  for (let i=1;i<N;i++) {
    ctx.beginPath();ctx.moveTo(i*sz,0);ctx.lineTo(i*sz,canvas.height);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,i*sz);ctx.lineTo(canvas.width,i*sz);ctx.stroke();
  }
  if (d.map_cell_m==null||d.odom_x==null) return;
  const ox=d.map_origin_x,oy=d.map_origin_y,cm=d.map_cell_m;
  function toCanvas(wx,wy){return[((wx-ox)/cm)*sz,(N-(wy-oy)/cm)*sz];}
  if (d.goal_odom_x!=null) {
    const [gx,gy]=toCanvas(d.goal_odom_x,d.goal_odom_y),r=sz*0.7;
    ctx.strokeStyle='#4af';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(gx-r,gy);ctx.lineTo(gx+r,gy);ctx.stroke();
    ctx.beginPath();ctx.moveTo(gx,gy-r);ctx.lineTo(gx,gy+r);ctx.stroke();
  }
  const [rx,ry]=toCanvas(d.odom_x,d.odom_y),rr=Math.max(sz*0.55,5);
  ctx.fillStyle='#e44'; ctx.beginPath();ctx.arc(rx,ry,rr,0,Math.PI*2);ctx.fill();
  if (d.odom_theta_deg!=null) {
    const th=d.odom_theta_deg*Math.PI/180;
    ctx.strokeStyle='#fa0';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(rx,ry);ctx.lineTo(rx+Math.cos(th)*rr*2.5,ry-Math.sin(th)*rr*2.5);ctx.stroke();
  }
}

connect();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helpers (from trajectory_test.py)
# ---------------------------------------------------------------------------

def _yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _yaw_to_quat(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SteamDeckWSTeleop(Node):

    IDLE = 'IDLE'
    RUNNING = 'RUNNING'

    def __init__(self):
        super().__init__('steamdeck_ws_teleop')

        # --- Parameters ---
        self.declare_parameter('ui_mode', 'phone')  # 'phone' or 'full'
        self.declare_parameter('ws_host', '0.0.0.0')
        self.declare_parameter('ws_port', 8888)
        self.declare_parameter('joy_rate_hz', 20.0)
        self.declare_parameter('map_push_rate_hz', 2.0)
        self.declare_parameter('coarse_map_size', 20)
        self.declare_parameter('goal_scale_m_per_tick', 0.02)
        self.declare_parameter('yaw_scale_rad_per_tick', 0.03)
        self.declare_parameter('max_goal_dist_m', 3.0)
        self.declare_parameter('dead_man_axis', 5)
        self.declare_parameter('dead_man_threshold', 0.5)
        self.declare_parameter('forward_axis', 1)
        self.declare_parameter('strafe_axis', 0)
        self.declare_parameter('yaw_axis', 3)
        self.declare_parameter('confirm_button', 0)
        self.declare_parameter('cancel_button', 1)
        self.declare_parameter('estop_button', 8)
        self.declare_parameter('disable_estop', False)  # TEMP: set true when estop topic not wired up
        self.declare_parameter('planner_action_name', '/compute_path_to_pose')
        self.declare_parameter('controller_action_name', '/follow_path')
        self.declare_parameter('planner_id', 'GridBased')
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('goal_checker_id', 'goal_checker')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('mode_set_topic', '/robot_mode_set')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('stuck_topic', '/robot_stuck')
        self.declare_parameter('kill_switch_topic', '/kill_switch')
        self.declare_parameter('lid_state_topic', '/lid_state')
        self.declare_parameter('preview_topic', '/steamdeck/goal_preview')
        self.declare_parameter('auto_set_driving_mode', True)
        self.declare_parameter('driving_mode_delay_s', 5.0)
        self.declare_parameter('trajectory_goal_topic',    '/trajectory_goal')
        self.declare_parameter('trajectory_cmd_topic',     '/trajectory_cmd')
        self.declare_parameter('lid_command_topic',        '/lid_command')
        self.declare_parameter('balance_gains_topic',      '/balance_gains')
        self.declare_parameter('balance_gains_echo_topic', '/balance_gains_echo')
        self.declare_parameter('driving_gains_topic',      '/driving_gains')
        self.declare_parameter('driving_gains_echo_topic', '/driving_gains_echo')
        self.declare_parameter('trajectory_status_topic',  '/trajectory_status')
        self.declare_parameter('leg_running_topic',         '/leg_controller_running')
        self.declare_parameter('user_valid_topic',          '/user_valid')
        self.declare_parameter('battery_voltage_topic',    '/vesc/battery_voltage')
        self.declare_parameter('cmd_vel_topic',            '/cmd_vel')
        self.declare_parameter('cmd_vel_safe_topic',       '/cmd_vel_safe')
        self.declare_parameter('driver_gains_echo_topic',  '/driver_gains_echo')
        self.declare_parameter('driver_gains_topic',       '/driver_gains')
        self.declare_parameter('vesc_gains_echo_topic',    '/vesc_gains_echo')
        self.declare_parameter('vesc_gains_topic',         '/vesc_gains')
        self.declare_parameter('set_pose_topic',           '/set_pose')

        p = self.get_parameter
        _ui_mode = str(p('ui_mode').value)
        self._html = _HTML_PHONE if _ui_mode == 'phone' else _HTML_FULL
        self._ws_host           = str(p('ws_host').value)
        self._ws_port           = int(p('ws_port').value)
        self._map_push_rate     = float(p('map_push_rate_hz').value)
        self._coarse_map_size   = int(p('coarse_map_size').value)
        self._goal_scale        = float(p('goal_scale_m_per_tick').value)
        self._yaw_scale         = float(p('yaw_scale_rad_per_tick').value)
        self._max_dist          = float(p('max_goal_dist_m').value)
        self._dm_axis           = int(p('dead_man_axis').value)
        self._dm_thresh         = float(p('dead_man_threshold').value)
        self._fwd_axis          = int(p('forward_axis').value)
        self._str_axis          = int(p('strafe_axis').value)
        self._yaw_axis          = int(p('yaw_axis').value)
        self._confirm_btn       = int(p('confirm_button').value)
        self._cancel_btn        = int(p('cancel_button').value)
        self._estop_btn         = int(p('estop_button').value)
        self._disable_estop     = bool(p('disable_estop').value)
        # --- State ---
        self._state = self.IDLE
        self._robot_mode = 'off'
        self._estop = False
        self._robot_stuck = False
        self._kill_switch = False
        self._lid_state = 'unknown'
        self._balance_gains_echo = ''
        self._traj_status = ''
        self._driver_gains_echo = ''
        self._vesc_gains_echo = ''
        self._driving_gains_echo = ''
        self._leg_running = False
        self._user_valid = False
        self._battery_voltage: float | None = None
        self._latest_cmd_vel: Twist | None = None
        self._latest_cmd_vel_safe: Twist | None = None
        self._latest_odom: Odometry | None = None
        self._latest_costmap: OccupancyGrid | None = None

        self._goal_x = 0.0
        self._goal_y = 0.0
        self._goal_theta = 0.0
        self._dead_man_prev = False
        self._prev_confirm = 0
        self._prev_cancel  = 0
        self._prev_estop   = 0

        # Shared joy state (written by WS thread, read by rclpy timer)
        self._lock = threading.Lock()
        self._joy_state = {'axes': [0.0] * 10, 'buttons': [0] * 12}

        # --- QoS ---
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_tl = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Subscriptions ---
        self.create_subscription(String,        p('mode_topic').value,        self._mode_cb,        reliable_tl)
        self.create_subscription(Odometry,      p('odom_topic').value,        self._odom_cb,        best_effort)
        self.create_subscription(Bool,          p('estop_topic').value,       self._estop_cb,       best_effort)
        self.create_subscription(OccupancyGrid, p('costmap_topic').value,     self._costmap_cb,     reliable_tl)
        self.create_subscription(Bool,          p('stuck_topic').value,       self._stuck_cb,       best_effort)
        self.create_subscription(Bool,          p('kill_switch_topic').value, self._kill_switch_cb, best_effort)
        self.create_subscription(String,        p('lid_state_topic').value,          self._lid_cb,                reliable)
        self.create_subscription(String,        p('balance_gains_echo_topic').value,  self._balance_gains_echo_cb,  reliable)
        self.create_subscription(String,        p('driving_gains_echo_topic').value,  self._driving_gains_echo_cb,  reliable)
        self.create_subscription(String,        p('trajectory_status_topic').value,  self._traj_status_cb,        reliable)
        self.create_subscription(Bool,          p('leg_running_topic').value,        self._leg_running_cb,        reliable_tl)
        self.create_subscription(Bool,          p('user_valid_topic').value,         self._user_valid_cb,         best_effort)
        self.create_subscription(Float64,       p('battery_voltage_topic').value,    self._battery_voltage_cb,    best_effort)
        self.create_subscription(Twist,         p('cmd_vel_topic').value,            self._cmd_vel_cb,            best_effort)
        self.create_subscription(Twist,         p('cmd_vel_safe_topic').value,       self._cmd_vel_safe_cb,       best_effort)
        self.create_subscription(String,        p('driver_gains_echo_topic').value,  self._driver_gains_echo_cb,  reliable)
        self.create_subscription(String,        p('vesc_gains_echo_topic').value,    self._vesc_gains_echo_cb,    reliable)

        # --- Publishers ---
        self._preview_pub      = self.create_publisher(PoseStamped, p('preview_topic').value,          10)
        self._mode_set_pub     = self.create_publisher(String,      p('mode_set_topic').value,         10)
        self._estop_pub        = self.create_publisher(Bool,        p('estop_topic').value,            10)
        self._traj_goal_pub    = self.create_publisher(String,      p('trajectory_goal_topic').value,  10)
        self._traj_cmd_pub     = self.create_publisher(String,      p('trajectory_cmd_topic').value,   10)
        self._lid_cmd_pub      = self.create_publisher(String,      p('lid_command_topic').value,      10)
        self._balance_gains_pub  = self.create_publisher(String,     p('balance_gains_topic').value,    10)
        self._driving_gains_pub  = self.create_publisher(String,     p('driving_gains_topic').value,    10)
        self._driver_gains_pub  = self.create_publisher(String,     p('driver_gains_topic').value,     10)
        self._vesc_gains_pub    = self.create_publisher(String,     p('vesc_gains_topic').value,       10)
        self._set_pose_pub      = self.create_publisher(PoseWithCovarianceStamped, p('set_pose_topic').value, 10)

        # --- Timers ---
        rate = float(p('joy_rate_hz').value)
        self.create_timer(1.0 / max(rate, 1.0), self._joy_tick)

        if p('auto_set_driving_mode').value:
            delay = float(p('driving_mode_delay_s').value)
            self._mode_timer = self.create_timer(delay, self._auto_set_driving_mode)

        # --- WebSocket server in background daemon thread ---
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

        self.get_logger().info(
            f'steamdeck_ws_teleop ready | ws://0.0.0.0:{self._ws_port} | '
            f'open http://ROBOT_IP:{self._ws_port} in Steam Deck browser'
        )

    # ------------------------------------------------------------------
    # WebSocket server (background daemon thread)
    # ------------------------------------------------------------------

    def _run_ws_server(self):
        asyncio.run(self._ws_main())

    async def _ws_main(self):
        async def process_request(connection, request):
            upgrade = request.headers.get('Upgrade', '').lower()
            if upgrade != 'websocket':
                body = self._html.encode()
                headers = WsHeaders([
                    ('Content-Type', 'text/html; charset=utf-8'),
                    ('Content-Length', str(len(body))),
                ])
                return WsResponse(200, 'OK', headers, body)

        async with ws_serve(
            self._ws_handler,
            self._ws_host,
            self._ws_port,
            process_request=process_request,
        ):
            self.get_logger().info(f'WebSocket server listening on port {self._ws_port}')
            await asyncio.Future()  # run forever

    async def _ws_handler(self, websocket):
        remote = websocket.remote_address
        self.get_logger().info(f'Steam Deck connected from {remote}')
        push_task = asyncio.create_task(self._push_loop(websocket))
        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if 'type' in data:
                    self._handle_panel_cmd(data)
                    continue
                with self._lock:
                    self._joy_state = data
                status = {
                    'type':          'status',
                    'state':         self._state,
                    'mode':          self._robot_mode,
                    'goal_x':        round(self._goal_x, 2),
                    'goal_y':        round(self._goal_y, 2),
                    'goal_theta_deg': round(math.degrees(self._goal_theta), 1),
                }
                await websocket.send(json.dumps(status))
        except websockets.ConnectionClosed:
            pass
        finally:
            push_task.cancel()
        self.get_logger().info(f'Steam Deck disconnected from {remote}')
        with self._lock:
            self._joy_state = {'axes': [0.0] * 10, 'buttons': [0] * 12}

    async def _push_loop(self, websocket):
        interval = 1.0 / max(self._map_push_rate, 0.1)
        while True:
            await asyncio.sleep(interval)
            try:
                await websocket.send(json.dumps(self._build_push_msg()))
            except websockets.ConnectionClosed:
                break

    def _build_push_msg(self) -> dict:
        odom = self._latest_odom
        odom_x = odom_y = odom_theta = None
        if odom is not None:
            odom_x     = round(odom.pose.pose.position.x, 2)
            odom_y     = round(odom.pose.pose.position.y, 2)
            odom_theta = round(math.degrees(_yaw_from_quat(odom.pose.pose.orientation)), 1)

        # Goal in odom frame (for map marker)
        goal_odom_x = goal_odom_y = None
        if odom is not None and (self._goal_x != 0.0 or self._goal_y != 0.0):
            g = self._compute_odom_goal()
            goal_odom_x = round(g.pose.position.x, 2)
            goal_odom_y = round(g.pose.position.y, 2)

        msg: dict = {
            'type':        'push',
            'mode':        self._robot_mode,
            'state':       self._state,
            'estop':       self._estop,
            'stuck':       self._robot_stuck,
            'kill_switch': self._kill_switch,
            'lid':         self._lid_state,
            'odom_x':      odom_x,
            'odom_y':      odom_y,
            'odom_theta_deg': odom_theta,
            'goal_odom_x': goal_odom_x,
            'goal_odom_y': goal_odom_y,
            'map':         self._encode_coarse_map(),
        }

        g = self._latest_costmap
        if g is not None:
            N = self._coarse_map_size
            msg['map_origin_x'] = round(g.info.origin.position.x, 3)
            msg['map_origin_y'] = round(g.info.origin.position.y, 3)
            msg['map_cell_m']   = round(g.info.resolution * g.info.width / N, 3)

        msg['planner_ready'] = '/compute_path_to_pose/_action/send_goal' in \
            [s for s, _ in self.get_service_names_and_types()]
        msg['legs_running']  = self._leg_running
        msg['user_valid']    = self._user_valid
        if self._battery_voltage is not None:
            msg['battery_v'] = round(self._battery_voltage, 1)

        if self._balance_gains_echo:
            msg['balance_gains'] = self._balance_gains_echo
        if self._driving_gains_echo:
            msg['driving_gains'] = self._driving_gains_echo
        if self._traj_status:
            msg['traj_status'] = self._traj_status
        if self._driver_gains_echo:
            msg['driver_gains'] = self._driver_gains_echo
        if self._vesc_gains_echo:
            msg['vesc_gains'] = self._vesc_gains_echo

        if self._latest_cmd_vel_safe is not None:
            t = self._latest_cmd_vel_safe
            msg['ctrl_in_vx'] = round(t.linear.x, 3)
            msg['ctrl_in_wz'] = round(t.angular.z, 3)
        if self._latest_cmd_vel is not None:
            t = self._latest_cmd_vel
            msg['motor_vx'] = round(t.linear.x, 3)
            msg['motor_wz'] = round(t.angular.z, 3)

        return msg

    def _encode_coarse_map(self) -> str | None:
        grid = self._latest_costmap
        if grid is None:
            return None
        N = self._coarse_map_size
        w, h = grid.info.width, grid.info.height
        if w < N or h < N:
            return None
        block_w = w // N
        block_h = h // N
        data = np.frombuffer(bytes(grid.data), dtype=np.int8).reshape((h, w))
        # Crop to exact multiple, then max-pool to NxN
        sub = data[:block_h * N, :block_w * N]
        coarse = sub.reshape(N, block_h, N, block_w).max(axis=(1, 3))
        # Encode: unknown=-1→'?', free=0→'0', occupied>0→'1'
        flat = coarse.ravel()
        result = bytearray(N * N)
        for i, v in enumerate(flat):
            result[i] = ord('?') if v < 0 else ord('0') if v == 0 else ord('1')
        return result.decode('ascii')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _mode_cb(self, msg: String):
        new_mode = msg.data.strip().lower()
        if new_mode == self._robot_mode:
            return
        prev = self._robot_mode
        self._robot_mode = new_mode
        self.get_logger().info(f'robot_mode: {prev} → {new_mode}')
        if self._state == self.RUNNING and new_mode != 'driving':
            self.get_logger().warn('Mode left "driving" — stopping navigation')
            self._stop()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg

    def _estop_cb(self, msg: Bool):
        if self._disable_estop:
            return
        new_val = bool(msg.data)
        if new_val and not self._estop:
            self.get_logger().warn('Estop received — stopping navigation')
            self._stop()
        self._estop = new_val

    def _costmap_cb(self, msg: OccupancyGrid):
        self._latest_costmap = msg

    def _stuck_cb(self, msg: Bool):
        self._robot_stuck = bool(msg.data)

    def _kill_switch_cb(self, msg: Bool):
        self._kill_switch = bool(msg.data)

    def _lid_cb(self, msg: String):
        self._lid_state = msg.data.strip()

    def _balance_gains_echo_cb(self, msg: String):
        self._balance_gains_echo = msg.data

    def _traj_status_cb(self, msg: String):
        self._traj_status = msg.data

    def _leg_running_cb(self, msg: Bool):
        self._leg_running = bool(msg.data)

    def _user_valid_cb(self, msg: Bool):
        self._user_valid = bool(msg.data)

    def _battery_voltage_cb(self, msg: Float64):
        self._battery_voltage = float(msg.data)

    def _cmd_vel_cb(self, msg: Twist):
        self._latest_cmd_vel = msg

    def _cmd_vel_safe_cb(self, msg: Twist):
        self._latest_cmd_vel_safe = msg

    def _driver_gains_echo_cb(self, msg: String):
        self._driver_gains_echo = msg.data

    def _vesc_gains_echo_cb(self, msg: String):
        self._vesc_gains_echo = msg.data

    def _driving_gains_echo_cb(self, msg: String):
        self._driving_gains_echo = msg.data

    # ------------------------------------------------------------------
    # Joy tick — runs at joy_rate_hz (20 Hz) via rclpy timer
    # ------------------------------------------------------------------

    def _joy_tick(self):
        with self._lock:
            joy = dict(self._joy_state)

        axes    = joy.get('axes', [])
        buttons = joy.get('buttons', [])

        def _axis(i: int) -> float:
            return axes[i] if i < len(axes) else 0.0

        def _btn(i: int) -> int:
            return buttons[i] if i < len(buttons) else 0

        # Dead-man (right trigger)
        dead_man = _axis(self._dm_axis) > self._dm_thresh

        # Falling edge: trigger released → reset goal accumulator
        if self._dead_man_prev and not dead_man:
            self._goal_x = self._goal_y = self._goal_theta = 0.0
        self._dead_man_prev = dead_man

        # Accumulate goal while armed
        if dead_man:
            self._goal_x = _clamp(
                self._goal_x + _axis(self._fwd_axis) * self._goal_scale,
                -self._max_dist, self._max_dist,
            )
            self._goal_y = _clamp(
                self._goal_y + _axis(self._str_axis) * self._goal_scale,
                -self._max_dist, self._max_dist,
            )
            self._goal_theta = _wrap(
                self._goal_theta + _axis(self._yaw_axis) * self._yaw_scale
            )

        # Button edge detection
        confirm = _btn(self._confirm_btn)
        cancel  = _btn(self._cancel_btn)
        estop   = _btn(self._estop_btn)

        if confirm == 1 and self._prev_confirm == 0:
            self._on_confirm_pressed()
        if cancel == 1 and self._prev_cancel == 0:
            self._on_cancel_pressed()
        if estop == 1 and self._prev_estop == 0:
            self._on_estop_pressed()

        self._prev_confirm = confirm
        self._prev_cancel  = cancel
        self._prev_estop   = estop

        # Publish goal preview for Foxglove
        if self._latest_odom is not None:
            self._preview_pub.publish(self._compute_odom_goal())

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_confirm_pressed(self):
        if self._robot_mode not in ('driving', 'balance'):
            self.get_logger().warn(f'A pressed but mode is "{self._robot_mode}" — need "driving" or "balance"')
            return
        if self._estop:
            self.get_logger().warn('A pressed but estop is active')
            return
        if self._latest_odom is None:
            self.get_logger().error('No odometry yet — cannot send goal')
            return

        goal_pose = self._compute_odom_goal()
        self.get_logger().info(
            f'Sending goal: ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})'
        )

        # Route through trajectory_test (avoids duplicate action clients)
        goal_msg = String()
        goal_msg.data = json.dumps({
            'x': goal_pose.pose.position.x,
            'y': goal_pose.pose.position.y,
            'theta': _yaw_from_quat(goal_pose.pose.orientation),
            'relative': False,
        })
        self._traj_goal_pub.publish(goal_msg)
        cmd_msg = String()
        cmd_msg.data = 'go'
        self._traj_cmd_pub.publish(cmd_msg)
        self._state = self.RUNNING

    def _on_cancel_pressed(self):
        if self._state == self.RUNNING:
            self.get_logger().info('B pressed — cancelling navigation')
            self._stop()
        else:
            self._goal_x = self._goal_y = self._goal_theta = 0.0
            self.get_logger().info('B pressed (idle) — goal accumulator reset')
            msg = String(); msg.data = 'reset'
            self._traj_cmd_pub.publish(msg)

    def _on_estop_pressed(self):
        self.get_logger().warn('Menu pressed — E-STOP')
        estop_msg = Bool(); estop_msg.data = True
        self._estop_pub.publish(estop_msg)
        mode_msg = String(); mode_msg.data = 'off'
        self._mode_set_pub.publish(mode_msg)
        self._stop()

    # ------------------------------------------------------------------
    # Frame math
    # ------------------------------------------------------------------

    def _compute_odom_goal(self) -> PoseStamped:
        odom = self._latest_odom
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        yaw = _yaw_from_quat(odom.pose.pose.orientation)

        gx     = rx + math.cos(yaw) * self._goal_x - math.sin(yaw) * self._goal_y
        gy     = ry + math.sin(yaw) * self._goal_x + math.cos(yaw) * self._goal_y
        gtheta = _wrap(yaw + self._goal_theta)

        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = gx
        pose.pose.position.y = gy
        pose.pose.orientation = _yaw_to_quat(gtheta)
        return pose

    # ------------------------------------------------------------------
    # Panel command handler (called from WS thread — publish is thread-safe)
    # ------------------------------------------------------------------

    def _handle_panel_cmd(self, data: dict):
        t = data.get('type')
        msg = String()
        if t == 'clear_estop':
            estop_msg = Bool(); estop_msg.data = False
            self._estop_pub.publish(estop_msg)
            self.get_logger().info('E-stop cleared via web UI.')
        elif t == 'estop':
            estop_msg = Bool(); estop_msg.data = True
            self._estop_pub.publish(estop_msg)
            msg.data = 'off'
            self._mode_set_pub.publish(msg)
        elif t == 'set_mode':
            msg.data = str(data.get('mode', 'off'))
            self._mode_set_pub.publish(msg)
        elif t == 'traj_goal':
            msg.data = json.dumps({
                'x':        float(data.get('x', 0.0)),
                'y':        float(data.get('y', 0.0)),
                'theta':    float(data.get('theta', 0.0)),
                'relative': bool(data.get('relative', True)),
            })
            self._traj_goal_pub.publish(msg)
        elif t == 'traj_cmd':
            msg.data = str(data.get('cmd', 'stop'))
            self._traj_cmd_pub.publish(msg)
        elif t == 'lid_cmd':
            msg.data = str(data.get('cmd', 'toggle'))
            self._lid_cmd_pub.publish(msg)
        elif t == 'balance_gains':
            msg.data = json.dumps(data.get('gains', {}))
            self._balance_gains_pub.publish(msg)
        elif t == 'driver_gains':
            msg.data = json.dumps(data.get('gains', {}))
            self._driver_gains_pub.publish(msg)
        elif t == 'vesc_gains':
            msg.data = json.dumps(data.get('gains', {}))
            self._vesc_gains_pub.publish(msg)
        elif t == 'driving_gains':
            msg.data = json.dumps(data.get('gains', {}))
            self._driving_gains_pub.publish(msg)
        elif t == 'reset_odom':
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.frame_id = 'odom'
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.pose.orientation.w = 1.0
            self._set_pose_pub.publish(pose_msg)
            self.get_logger().info('Odometry reset to origin.')

    # ------------------------------------------------------------------
    # Auto mode setter
    # ------------------------------------------------------------------

    def _auto_set_driving_mode(self):
        self._mode_timer.cancel()
        msg = String(); msg.data = 'driving'
        self._mode_set_pub.publish(msg)
        self.get_logger().info('Auto-set robot mode to "driving"')

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def _stop(self):
        self._state = self.IDLE
        msg = String(); msg.data = 'stop'
        self._traj_cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = SteamDeckWSTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
