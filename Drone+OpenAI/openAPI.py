from openai import OpenAI
import json

client = OpenAI(
    api_key=""
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
    }
]

def call_function(name, args):
    if name == "go_up":
        return go_up(**args)
    if name == "go_down":
        return go_down(**args)

def go_up(centimeters):
    print(f"🛸 Going up {centimeters} cm")

def go_down(centimeters):
    print(f"🛸 Going down {centimeters} cm")

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
            args = json.loads(tool_call.function.arguments)
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
