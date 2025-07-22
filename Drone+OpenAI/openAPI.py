from openai import OpenAI
from drone_controller import DroneController
import json, threading

dc = DroneController()
dc.start()
maxNumCommands = 5

client = OpenAI(
    api_key=""  # <-- Insert your API key here
)

tools = [
    {
        "name": "go_up",
        "description": "Drone moves up x centimeters",
        "parameters": {
            "type": "object",
            "properties": {
                "centimeters": {"type": "number"}
            },
            "required": ["centimeters"]
        }
    },
    {
        "name": "go_down",
        "description": "Drone moves down x centimeters",
        "parameters": {
            "type": "object",
            "properties": {
                "centimeters": {"type": "number"}
            },
            "required": ["centimeters"]
        }
    },
    {
        "name": "go_forward",
        "description": "Drone moves forward x centimeters",
        "parameters": {
            "type": "object",
            "properties": {
                "centimeters": {"type": "number"}
            },
            "required": ["centimeters"]
        }
    },
    {
        "name": "go_back",
        "description": "Drone moves backward x centimeters",
        "parameters": {
            "type": "object",
            "properties": {
                "centimeters": {"type": "number"}
            },
            "required": ["centimeters"]
        }
    },
    {
        "name": "land",
        "description": "Drone initiates landing"
    },
    {
        "name": "takeoff",
        "description": "Drone initiates takeoff"
    },
    {
        "name": "rotate_clockwise",
        "description": "Drone rotates clockwise x degrees",
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {"type": "number"}
            },
            "required": ["degrees"]
        }
    },
    {
        "name": "rotate_counterclockwise",
        "description": "Drone rotates counterclockwise x degrees",
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {"type": "number"}
            },
            "required": ["degrees"]
        }
    },
    {
        "name": "follow_target_sequence",
        "description": "Provide a target description which will allow a vision agent to set a bounding box around a target the drone will then follow",
        "parameters": {
            "type": "object",
            "properties": {
                "target_description": {"type": "string"}
            },
            "required": ["target_description"]
        }
    },
    {
        "name": "start",
        "description": "Start up the drone and begin streaming "
    }
]
def _can_enqueue(cmd):
    if dc.cmd_gate:
        print("🚧 Landing in progress; ignoring new commands.")
        return False
    with dc.cmd_q.mutex:
        if dc.cmd_q.qsize() >= maxNumCommands:
            print("⚠️ Queue full."); return False
        if dc.cmd_q.queue and dc.cmd_q.queue[-1] == cmd:
            print("⚠️ Duplicate last command; skipped."); return False
    return True


def go_up(centimeters):
    cmd = f"up {centimeters}"
    if _can_enqueue(cmd):
        dc.cmd_q.put(cmd)
        print(f"🛸 Going up {centimeters} cm")

def go_down(centimeters):
    cmd = f"down {centimeters}"
    if _can_enqueue(cmd):
        dc.cmd_q.put(cmd)
        print(f"🛸 Going down {centimeters} cm")

def go_forward(centimeters):
    cmd = f"forward {centimeters}"
    if _can_enqueue(cmd):
        dc.cmd_q.put(cmd)
        print(f"🛸 Moving forward {centimeters} cm")

def go_back(centimeters):
    cmd = f"back {centimeters}"
    if _can_enqueue(cmd):
        dc.cmd_q.put(cmd)
        print(f"🛸 Moving backward {centimeters} cm")

def takeoff():
    if dc.is_flying:
        print("⚠️ Already in the air."); return
    else:
        dc.cmd_q.put("takeoff")
        print("🛫 Take-off requested") 
        dc.cmd_gate = False # reopen pipeline

def land():
    if not dc.is_flying:
        print("⚠️ Already landed."); return

    if _can_enqueue("land"):
        dc.cmd_q.put("land")
        dc.cmd_gate = True           # close pipeline until next take‑off
        print("🛬 Landing requested")

def rotate_clockwise(degrees):
    cmd = f"cw {degrees}"
    if _can_enqueue(cmd):
        dc.cmd_q.put(cmd)
        print(f"🔁 Rotating clockwise {degrees}°")

def rotate_counterclockwise(degrees):
    cmd = f"ccw {degrees}"
    if _can_enqueue(cmd):
        dc.cmd_q.put(cmd)
        print(f"🔁 Rotating counterclockwise {degrees}°")

def follow_target_sequence(target_description):
    if not dc.is_flying:
        print("⚠️ Cannot follow target, drone is not flying.")
        return
    dc.follow(target_description)

def stop_follow_sequence():
    dc.stop_follow()

def start_drone():
    dc.start()

    
def call_function(name, args=None):
    if name == "go_up":
        return go_up(**args)
    elif name == "go_down":
        return go_down(**args)
    elif name == "go_forward":
        return go_forward(**args)
    elif name == "go_back":
        return go_back(**args)
    elif name == "takeoff":
        return takeoff()
    elif name == "land":
        return land()
    elif name == "rotate_clockwise":
        return rotate_clockwise(**args)
    elif name == "rotate_counterclockwise":
        return rotate_counterclockwise(**args)
    elif name == "follow_target_sequence":
        return follow_target_sequence(**args)
    elif name == "start":
        return start_drone(**args)
    else:
        print(f"⚠️ Unknown function call: {name}")

def promptgpt(inp):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a drone copilot that maps user input to drone movement functions. Every command you send is sent into a queue."},
            {"role": "user", "content": inp}
        ],
        tools=[{"type": "function", "function": tool} for tool in tools],
        tool_choice="auto"
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else None
            call_function(name, args)
    elif msg.content:
        print("🤖 GPT says:", msg.content)

# Main loop
if __name__ == "__main__":
    try:
        while True:
            user_input = input("Prompt> ")
            if user_input.lower() in ["exit", "quit"]:
                print("🛬 Exiting... initiating immediate landing.")
                dc.immediate_land()
                break
            promptgpt(user_input)
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt detected. Landing drone...")
        dc.immediate_land()

