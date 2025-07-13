# generator.py using Hugging Face FLAN-T5 with improved prompt

from transformers import pipeline

# Load FLAN-T5 model (base or large)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")  # Change to 'flan-t5-large' if needed

def generate_answer(question: str, context: str) -> str:
    """
    Generates an answer using Hugging Face FLAN-T5 model with enhanced prompt.
    """

    prompt = (
        f"Here are multiple cases of loan applicants and whether their loans were approved.\n\n"
        f"Context:\n{context}\n\n"
        f"Based on these cases, answer this question clearly and concisely:\n\n"
        f"{question}\n\n"
        f"Answer:"
    )

    try:
        response = qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"]
        return response.strip()
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"
