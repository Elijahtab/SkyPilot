#!/usr/bin/env python3
# openapi.py

import json
import threading
import signal
import sys
import time
import os

from openai import OpenAI
from drone_controller import DroneController

# ---- init globals ----
dc = DroneController()
client = OpenAI()  # needs OPENAI_API_KEY in env

# ---- tool schema (must match arg names you accept) ----
tools = [
    {"name": "go_up", "description": "Drone moves up x centimeters",
     "parameters": {"type": "object", "properties": {"centimeters": {"type": "number"}}, "required": ["centimeters"]}},
    {"name": "go_down", "description": "Drone moves down x centimeters",
     "parameters": {"type": "object", "properties": {"centimeters": {"type": "number"}}, "required": ["centimeters"]}},
    {"name": "go_forward", "description": "Drone moves forward x centimeters",
     "parameters": {"type": "object", "properties": {"centimeters": {"type": "number"}}, "required": ["centimeters"]}},
    {"name": "go_back", "description": "Drone moves backward x centimeters",
     "parameters": {"type": "object", "properties": {"centimeters": {"type": "number"}}, "required": ["centimeters"]}},
    {"name": "go_left", "description": "Drone moves left x centimeters",
     "parameters": {"type": "object", "properties": {"centimeters": {"type": "number"}}, "required": ["centimeters"]}},
    {"name": "go_right", "description": "Drone moves right x centimeters",
     "parameters": {"type": "object", "properties": {"centimeters": {"type": "number"}}, "required": ["centimeters"]}},
    {"name": "rotate_clockwise", "description": "Rotate clockwise by degrees",
     "parameters": {"type": "object", "properties": {"degrees": {"type": "number"}}, "required": ["degrees"]}},
    {"name": "rotate_counterclockwise", "description": "Rotate counter-clockwise by degrees",
     "parameters": {"type": "object", "properties": {"degrees": {"type": "number"}}, "required": ["degrees"]}},
    {"name": "flip", "description": "Flip in direction (l,r,f,b)",
     "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}, "required": ["direction"]}},
    {"name": "takeoff", "description": "Takeoff the drone"},
    {"name": "land", "description": "Land the drone normally"},
    {"name": "emergency_land", "description": "Immediate safe land (NOT motor kill)"},
    {"name": "follow_target_sequence", "description": "Follow a target described in text",
     "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
    {"name": "stop_follow_sequence", "description": "Stop following the current target"},
    {"name": "start", "description": "Start controller + video stream"},
    {"name": "stop", "description": "Stop controller + threads"},
]

# ---- wrappers (accept **kwargs to avoid TypeError if schema changes) ----
def _mv(cmd):
    def f(centimeters=None, **_):
        dc.enqueue(f"{cmd} {int(centimeters)}")
    return f

go_up      = _mv("up")
go_down    = _mv("down")
go_left    = _mv("left")
go_right   = _mv("right")
go_forward = _mv("forward")
go_back    = _mv("back")

def rotate_clockwise(degrees=None, **_):        dc.enqueue(f"cw {int(degrees)}")
def rotate_counterclockwise(degrees=None, **_): dc.enqueue(f"ccw {int(degrees)}")
def flip(direction=None, **_):                  dc.enqueue(f"flip {direction}")

def takeoff(**_):                 dc.takeoff()
def land(**_):                    dc.land()
def emergency_land(**_):          dc.immediate_land()          # soft land, not motor kill
def follow_target_sequence(description=None, **_): dc.start_follow(description or "")
def stop_follow_sequence(**_):    dc.stop_follow()
def start(**_):                   dc.start()
def stop(**_):                    dc.stop()

DISPATCH = {
    "go_up": go_up,
    "go_down": go_down,
    "go_forward": go_forward,
    "go_back": go_back,
    "go_left": go_left,
    "go_right": go_right,
    "rotate_clockwise": rotate_clockwise,
    "rotate_counterclockwise": rotate_counterclockwise,
    "flip": flip,
    "takeoff": takeoff,
    "land": land,
    "emergency_land": emergency_land,
    "follow_target_sequence": follow_target_sequence,
    "stop_follow_sequence": stop_follow_sequence,
    "start": start,
    "stop": stop,
}

def call_function(name, args=None):
    func = DISPATCH.get(name)
    if not func:
        print(f"⚠️ Unknown function call: {name}")
        return
    return func(**(args or {}))

# ---- GPT wrapper ----
def promptgpt(inp: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": inp}],
        tools=[{"type": "function", "function": t} for t in tools],
        tool_choice="auto"
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else None
            call_function(name, args)
    elif msg.content:
        print("🤖", msg.content)

# ---- simple REPL (background thread) ----
def repl():
    while True:
        try:
            user_input = input("Prompt> ").strip()
        except EOFError:
            break
        if user_input.lower() in {"quit", "exit"}:
            print("🛬 Exiting… soft land.")
            dc.immediate_land()
            dc.stop()
            return
        try:
            promptgpt(user_input)
        except Exception as e:
            print("REPL error:", e)

# ---- SIGINT (Ctrl‑C) handler on main thread ----
def sigint_handler(sig, frame):
    print("\n🛑 SIGINT -> soft land")
    dc.immediate_land()
    # allow a moment for 'ok' to arrive
    time.sleep(0.5)
    dc.stop()
    os._exit(130)

# ---- main ----
if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)

    t = threading.Thread(target=repl, daemon=True, name="repl")
    t.start()
    try:
        # Must be on main thread for macOS OpenCV
        dc.viewer_mainloop()
    finally:
        dc.stop()
        t.join(timeout=1)
        sys.exit(0)
