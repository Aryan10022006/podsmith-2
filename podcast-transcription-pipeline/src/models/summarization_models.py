from transformers import pipeline

class SummarizationModel:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_text(self, text, min_length=30, max_length=130):
        summary = self.summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
        return summary[0]['summary_text']

    def summarize_blocks(self, blocks):
        summaries = []
        for block in blocks:
            summary = self.summarize_text(block['text'])
            summaries.append({
                'block_id': block['block_id'],
                'summary': summary
            })
        return summaries

    def generate_global_summary(self, semantic_blocks):
        combined_text = " ".join([block['text'] for block in semantic_blocks])
        return self.summarize_text(combined_text)