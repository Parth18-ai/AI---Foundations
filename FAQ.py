"""
FAQ similarity search using embeddings + numpy (no vector database).

Run:
    python FAQ.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

faqs = [
    "How do I reset my password?",
    "How can I change my email address?",
    "What are your working hours?",
    "How do I contact customer support?",
    "How can I track my order?",
    "What is your refund policy?",
    "How long does delivery take?",
    "Can I cancel my order after placing it?",
    "Which payment methods do you accept?",
    "How do I create a new account?",
    "How do I delete my account?",
    "Is my personal data safe with you?",
    "How do I update my billing address?",
    "Do you ship to other countries?",
    "How can I apply a discount code?",
    "What should I do if I receive a damaged product?",
    "How do I download my invoice?",
    "Do you offer a student discount?",
    "How can I subscribe to your newsletter?",
    "How do I turn on two-factor authentication?",
    "How do I connect to the office Wi-Fi?",
]


print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


faq_embeddings = model.encode(faqs, normalize_embeddings=True)
print(f"Embedded {len(faqs)} FAQs. Shape: {faq_embeddings.shape}")


def search(question, top_k=3):
    """Return the top_k most similar FAQs for the given question."""

    question_embedding = model.encode(question, normalize_embeddings=True)

    scores = faq_embeddings @ question_embedding

    best_positions = np.argsort(scores)[::-1][:top_k]

    results = []
    for pos in best_positions:
        results.append((faqs[pos], float(scores[pos])))
    return results


if __name__ == "__main__":
    print("\nAsk me anything! (type 'quit' to stop)\n")

    while True:
        user_question = input("Your question: ").strip()

        if user_question.lower() == "quit":
            print("Bye!")
            break
        if not user_question:
            continue

        print("\nTop 3 matching FAQs:")
        for rank, (faq, score) in enumerate(search(user_question), start=1):
            print(f"  {rank}. {faq}  (score: {score:.2f})")
        print()
