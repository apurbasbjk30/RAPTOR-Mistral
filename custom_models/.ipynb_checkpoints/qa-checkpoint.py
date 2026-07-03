import ollama
from raptor.QAModels import BaseQAModel


class MistralQA(BaseQAModel):

    def answer_question(self, context, question):

        prompt = f"""
You are answering questions from a scientific paper.

Context:
{context}

Question:
{question}

Give a concise answer using only the context.
"""

        response = ollama.chat(
            model="mistral",
            keep_alive="30m",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]