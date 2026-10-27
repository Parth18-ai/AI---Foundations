import sys
import ollama

MODEL = "llama3.2:1b"


def ask(messages):
    """Send messages to the model and return its reply text."""
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]


def zero_shot(question):
    # Style 1: just ask directly, no examples
    return [
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user", "content": question},
    ]


def few_shot(question):
    # Style 2: show examples first, so the AI copies the style
    return [
        {"role": "system", "content": "You are a helpful teacher. Answer in exactly 2 short lines."},
        {"role": "user", "content": "What is a variable in Python?"},
        {"role": "assistant", "content": "A variable is a name that stores a value.\nExample: age = 20 stores the number 20 in the name age."},
        {"role": "user", "content": "What is a loop in Python?"},
        {"role": "assistant", "content": "A loop repeats the same code many times.\nExample: for i in range(3): print(i) prints 0, 1, 2."},
        {"role": "user", "content": question},
    ]


def chain_of_thought(question):
    # Style 3: ask the AI to think step by step
    return [
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user", "content": question + "\n\nThink step by step, then give the final answer."},
    ]


def main():
    # Get the question from the command line, or ask for it
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    styles = [
        ("ZERO-SHOT", zero_shot),
        ("FEW-SHOT", few_shot),
        ("CHAIN-OF-THOUGHT", chain_of_thought),
    ]

    print(f"\nQuestion: {question}\n")

    for name, builder in styles:
        print("=" * 60)
        print(f" {name}")
        print("=" * 60)
        try:
            print(ask(builder(question)))
        except Exception as error:
            print("Error:", error)
            print("Is Ollama running? Did you run 'ollama pull llama3.2:1b'?")
            return
        print()


if __name__ == "__main__":
    main()