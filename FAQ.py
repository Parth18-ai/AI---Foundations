"""
FAQ similarity search using embeddings + numpy (no vector database).

Run:
    python FAQ.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# STEP 1: Our FAQ questions (this is our "document list")
# Rule: every line ends with a comma, and nothing goes after the ]
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# STEP 2: Load the embedding model (the "teacher" who gives marks)
# The first run downloads it (~90 MB). After that it works offline.
# ---------------------------------------------------------------
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------
# STEP 3: Embed all FAQs ONCE and keep them in memory.
# normalize_embeddings=True makes every arrow length 1.
# When arrows have length 1, cosine similarity = simple dot product.
# Result shape: (number of FAQs, 384)
# ---------------------------------------------------------------
faq_embeddings = model.encode(faqs, normalize_embeddings=True)
print(f"Embedded {len(faqs)} FAQs. Shape: {faq_embeddings.shape}")


# ---------------------------------------------------------------
# STEP 4: The search function
# ---------------------------------------------------------------
def search(question, top_k=3):
    """Return the top_k most similar FAQs for the given question."""

    # a) Turn the question into numbers (same model, same settings!)
    question_embedding = model.encode(question, normalize_embeddings=True)

    # b) Compare the question with ALL FAQs at once (one score each)
    scores = faq_embeddings @ question_embedding

    # c) Positions of the highest scores.
    #    argsort sorts low -> high, so [::-1] flips it to high -> low.
    best_positions = np.argsort(scores)[::-1][:top_k]

    # d) Build a clean result list
    results = []
    for pos in best_positions:
        results.append((faqs[pos], float(scores[pos])))
    return results


# ---------------------------------------------------------------
# STEP 5: Let the user type questions
# ---------------------------------------------------------------
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