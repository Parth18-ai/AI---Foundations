
"""AutoExpert: a multi-turn CLI chatbot with a custom persona.

Features
  * Custom persona (system prompt) -- easy to edit below
  * Multi-turn conversation history (the model remembers earlier messages)
  * Save / load a session to a JSON file
  * Streaming replies (text appears as it is generated)
  * Runs locally and free using Ollama

Run:  python chatbot.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import ollama


DEFAULT_MODEL = "llama3.2"   
SESSIONS_DIR = Path("sessions")
MAX_MESSAGES = 40         

PERSONA_NAME = "AutoExpert"
SYSTEM_PROMPT = (
    "You are AutoExpert, a friendly and knowledgeable automobile expert with "
    "20 years of experience as a mechanic and car consultant. "
    "You help people with car buying advice, maintenance, troubleshooting "
    "warning lights and strange noises, fuel economy, and EV basics. "
    "Rules: explain things in simple language, give step-by-step advice when "
    "useful, keep answers under 150 words unless asked for more, and always "
    "recommend a professional mechanic for anything safety-critical (brakes, "
    "steering, airbags). If a question is not about cars, politely steer the "
    "conversation back to cars."
)

HELP_TEXT = """Commands:
  /help            show this help
  /save [name]     save the conversation to sessions/<name>.json
  /load <name>     load a saved conversation
  /list            list saved sessions
  /history         show the conversation so far
  /clear           start a fresh conversation
  /quit            exit (Ctrl+C also works)"""


def resolve_path(name):
    """Turn a short name like 'chat1' into sessions/chat1.json."""
    path = Path(name)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    if path.parent == Path("."):
        path = SESSIONS_DIR / path
    return path


def save_session(history, model, name=None):
    """Write the conversation to a JSON file and return the path used."""
    if not name:
        name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    path = resolve_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "persona": PERSONA_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "model": model,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": history,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_session(name):
    """Read a saved conversation. Returns (messages, model). Raises ValueError if bad."""
    path = resolve_path(name)
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"{path} is not valid JSON ({err})")


    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{path} has no 'messages' list")
    for m in messages:
        if not (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)):
            raise ValueError(f"{path} contains a badly formed message: {m!r}")
    return messages, data.get("model")


def list_sessions():
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))

def stream_reply(model, history):
    """Send the history to Ollama, print the reply as it streams, return it."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-MAX_MESSAGES:]

    reply = ""
    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        reply += piece
    print()
    return reply


def show_history(history):
    if not history:
        print("(no messages yet)")
        return
    for m in history:
        who = "You" if m["role"] == "user" else PERSONA_NAME
        print(f"{who}: {m['content']}\n")


def handle_command(line, history, model):
    """Handle a /command. Returns (history, model, keep_running)."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return history, model, False
    elif cmd == "/help":
        print(HELP_TEXT)
    elif cmd == "/save":
        path = save_session(history, model, arg or None)
        print(f"Saved to {path}")
    elif cmd == "/load":
        if not arg:
            print("Usage: /load <name>   (try /list to see saved sessions)")
        else:
            try:
                history, saved_model = load_session(arg)
                model = saved_model or model
                print(f"Loaded {len(history)} messages. Model: {model}")
            except ValueError as err:
                print(f"Could not load: {err}")
    elif cmd == "/list":
        names = list_sessions()
        print("\n".join(names) if names else "(no saved sessions yet)")
    elif cmd == "/history":
        show_history(history)
    elif cmd == "/clear":
        history = []
        print("Conversation cleared.")
    else:
        print("Unknown command. Type /help to see the options.")
    return history, model, True


def main():
    parser = argparse.ArgumentParser(description=f"{PERSONA_NAME} CLI chatbot")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--load", metavar="NAME",
                        help="load a saved session on startup")
    args = parser.parse_args()

    history = []
    model = args.model

    if args.load:
        try:
            history, saved_model = load_session(args.load)
            model = saved_model or model
            print(f"Loaded {len(history)} messages from {args.load}")
        except ValueError as err:
            print(f"Could not load: {err}")
            sys.exit(1)

    print(f"=== {PERSONA_NAME} === (model: {model})")
    print("Ask me anything about cars. Type /help for commands.\n")
    if history:
        show_history(history[-4:])  

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        if line.startswith("/"):
            history, model, keep_going = handle_command(line, history, model)
            if not keep_going:
                print("Bye!")
                break
            continue

        history.append({"role": "user", "content": line})
        print(f"{PERSONA_NAME}: ", end="", flush=True)
        try:
            reply = stream_reply(model, history)
        except KeyboardInterrupt:
            history.pop()
            print("\n(interrupted)")
            continue
        except Exception as err:  
            history.pop()  
            print(f"\n[Error talking to Ollama: {err}]")
            print(f"Check that the Ollama app is running and that the model "
                  f"is installed:  ollama pull {model}")
            continue

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()