from openai import OpenAI
from drone_controller import DroneController
import json

dc = DroneController()
client = OpenAI()  # set OPENAI_API_KEY in env

# ---------------- tool schema ----------------
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
    {"name": "emergency_land", "description": "Immediate land, bypassing queue"},
    {"name": "follow_target_sequence", "description": "Follow a target described in text",
     "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
    {"name": "stop_follow_sequence", "description": "Stop following the current target"},
    {"name": "start", "description": "Start controller + video stream"},
    {"name": "stop", "description": "Stop controller + threads"},
]

# ---------------- command helpers ----------------
def _mv(cmd):
    return lambda centimeters: dc.enqueue(f"{cmd} {int(centimeters)}")

go_up     = _mv("up")
go_down   = _mv("down")
go_left   = _mv("left")
go_right  = _mv("right")
go_forward= _mv("forward")
go_back   = _mv("back")

def rotate_clockwise(degrees):        dc.enqueue(f"cw {int(degrees)}")
def rotate_counterclockwise(degrees): dc.enqueue(f"ccw {int(degrees)}")
def flip(direction):                  dc.enqueue(f"flip {direction}")

def takeoff():                        dc.takeoff()
def land():                           dc.land()
def emergency_land():                 dc.immediate_land()
def follow_target_sequence(description): dc.start_follow(description)
def stop_follow_sequence():           dc.stop_follow()
def start():                          dc.start()
def stop():                           dc.stop()

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
    if args is None:
        return func()
    return func(**args)

# ---------------- GPT loop ----------------
def promptgpt(inp: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": inp}],
        tools=[{"type": "function", "function": t} for t in tools],
        tool_choice="auto"
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else None
            call_function(name, args)
    elif msg.content:
        print("🤖", msg.content)

# ---------------- shell ----------------
if __name__ == "__main__":
    try:
        while True:
            user_input = input("Prompt> ")
            if user_input.lower() in {"exit", "quit"}:
                print("🛬 Exiting… immediate landing.")
                dc.immediate_land()
                break
            promptgpt(user_input)
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt. Landing…")
        dc.immediate_land()
