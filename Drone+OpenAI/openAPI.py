from openai import OpenAI
import json

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
    }
]

def go_up(centimeters):
    print(f"🛸 Going up {centimeters} cm")

def go_down(centimeters):
    print(f"🛸 Going down {centimeters} cm")

def go_forward(centimeters):
    print(f"🛸 Moving forward {centimeters} cm")

def go_back(centimeters):
    print(f"🛸 Moving backward {centimeters} cm")

def takeoff():
    print("🛫 Taking off")

def land():
    print("🛬 Landing")

def rotate_clockwise(degrees):
    print(f"🔁 Rotating clockwise {degrees}°")

def rotate_counterclockwise(degrees):
    print(f"🔁 Rotating counterclockwise {degrees}°")

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
    else:
        print(f"⚠️ Unknown function call: {name}")

def promptgpt(inp):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a drone copilot that maps user input to drone movement functions."},
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
    while True:
        user_input = input("Prompt> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        promptgpt(user_input)
