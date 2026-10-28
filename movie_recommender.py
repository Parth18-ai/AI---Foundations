import argparse
import json
import textwrap

import ollama

DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a movie recommendation expert.
Reply with ONLY valid JSON, in exactly this shape:
{"recommendations": [
  {"title": "Movie name", "year": 2010, "genre": "Genre", "why": "One short sentence."}
]}
Rules:
- Give exactly 3 real movies.
- "year" must be a number.
- "why" must be one sentence, under 25 words.
- No text outside the JSON."""


def get_recommendations(model, user_input, tries=2):
    for attempt in range(1, tries + 1):
        response = ollama.chat(
            model=model,
            format="json",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Recommend movies for: {user_input}"},
            ],
        )
        raw_text = response["message"]["content"]
        try:
            data = json.loads(raw_text)
            recs = data["recommendations"] if isinstance(data, dict) else data
            if isinstance(recs, list) and recs:
                return recs[:3], raw_text
        except (json.JSONDecodeError, KeyError):
            pass
        print(f"(Attempt {attempt}: reply was not usable JSON, retrying...)")
    return [], ""


def show_recommendations(recs, user_input):
    line = "=" * 60
    print(f"\n{line}")
    print(f" Movie picks for: {user_input}")
    print(line)
    for number, movie in enumerate(recs, start=1):
        title = movie.get("title", "Unknown title")
        year = movie.get("year", "?")
        genre = movie.get("genre", "?")
        why = movie.get("why", "")
        print(f"\n {number}. {title} ({year})")
        print(f"    Genre: {genre}")
        print("    Why:   " + textwrap.fill(str(why), width=50,
                                           subsequent_indent="           "))
    print(f"\n{line}\n")


def main():
    parser = argparse.ArgumentParser(description="Get 3 movie recommendations.")
    parser.add_argument("mood", nargs="?", help='e.g. "feel-good comedy"')
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--json", action="store_true", help="Also print raw JSON")
    args = parser.parse_args()

    mood = args.mood or input("What genre or mood are you in for? ")

    print("Thinking... (local models can take 10-60 seconds)")
    try:
        recs, raw_json = get_recommendations(args.model, mood)
    except Exception as err:
        print(f"Error: {err}")
        print(f"Is Ollama running, and did you run: ollama pull {args.model}")
        return

    if not recs:
        print("Sorry, the model did not return usable results. Try again.")
        return

    show_recommendations(recs, mood)

    if args.json:
        print("Raw JSON from the model:")
        print(json.dumps(json.loads(raw_json), indent=2))


if __name__ == "__main__":
    main()