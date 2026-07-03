import ollama
from raptor.SummarizationModels import BaseSummarizationModel


class MistralSummarizer(BaseSummarizationModel):

    def summarize(self, context, max_tokens=150):

        prompt = f"""
Summarize the following scientific text while preserving important technical information.

{context}
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