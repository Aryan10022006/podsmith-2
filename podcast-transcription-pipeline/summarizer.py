import json
import logging
import sys
import time
from ollama import Client

OLLAMA_MODEL = "mistral"  # Change to your preferred Ollama model
SEMANTIC_BLOCKS_PATH = "semantic_blocks.json"
OUTPUT_SUMMARIES_PATH = "summaries.json"

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("summarizer")

def load_blocks(path):
    logger.info(f"Loading semantic blocks from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ollama_call(client, prompt, system=None):
    logger.info(f"LLM INPUT PROMPT:\n{prompt}\n")
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt, system=system)
    output = response['response'].strip()
    logger.info(f"LLM OUTPUT:\n{output}\n")
    return output

def extract_topic(client, text):
    prompt = (
        "You are an expert content analyst. "
        "Extract a concise, domain-relevant topic for the following content. "
        "Return only the topic phrase.\n\n"
        f"CONTENT:\n{text}"
    )
    return ollama_call(client, prompt)

def generate_summary(client, text, topic):
    prompt = (
        f"You are a professional summarizer. Summarize the following content with a focus on the topic '{topic}'. "
        "Make the summary concise, coherent, and domain-specific. Do not copy the original text, but synthesize the main ideas and insights.\n\n"
        f"CONTENT:\n{text}"
    )
    return ollama_call(client, prompt)

def extract_themes(client, text, topic):
    prompt = (
        f"You are a domain expert. List 2-4 main themes present in the following content about '{topic}'. "
        "Return only the themes as a bullet list.\n\n"
        f"CONTENT:\n{text}"
    )
    output = ollama_call(client, prompt)
    themes = [line.lstrip("-•* ").strip() for line in output.splitlines() if line.strip()]
    return themes

def extract_key_points(client, text, topic):
    prompt = (
        f"You are a semantic extractor. Identify 3-5 key points from the following content about '{topic}'. "
        "Return only the key points as a bullet list.\n\n"
        f"CONTENT:\n{text}"
    )
    output = ollama_call(client, prompt)
    key_points = [line.lstrip("-•* ").strip() for line in output.splitlines() if line.strip()]
    return key_points

def main():
    start_time = time.time()
    client = Client()
    blocks = load_blocks(SEMANTIC_BLOCKS_PATH)

    # --- Global topic and summary ---
    logger.info("Generating global topic and summary...")
    all_text = " ".join(block["text"] for block in blocks)
    global_topic = extract_topic(client, all_text)
    global_summary = generate_summary(client, all_text, global_topic)

    # --- Blockwise summaries ---
    block_outputs = []
    for block in blocks:
        block_text = block["text"]
        block_topic = extract_topic(client, block_text)
        block_summary = generate_summary(client, block_text, block_topic)
        themes = extract_themes(client, block_text, block_topic)
        key_points = extract_key_points(client, block_text, block_topic)
        block_outputs.append({
            "block_id": block["block_id"],
            "topic": block_topic,
            "summary": block_summary,
            "themes": themes,
            "key_points": key_points
        })
        logger.info(f"Block {block['block_id']} processed.")

    output = {
        "global_summary": global_summary,
        "global_topic": global_topic,
        "blocks": block_outputs
    }
    logger.info(f"Saving output to {OUTPUT_SUMMARIES_PATH}")
    with open(OUTPUT_SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    total_time = time.time() - start_time
    logger.info(f"Summarization complete. Total time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()