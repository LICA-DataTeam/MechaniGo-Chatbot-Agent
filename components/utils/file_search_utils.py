# Documentation/Guide: https://platform.openai.com/docs/guides/tools-file-search
from dotenv import load_dotenv
from openai import OpenAI
import logging
import json
import os

# IMPORTANT:
# - This script only needs to be run once (or again when FAQ file changes) to save FAQs to vector store
# - Once saved, the FAQ data is indexed, OpenAI agent can query it now

# How to run from root directory:
# Enter python -m components.utils.file_search_utils

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_file(client: OpenAI = None, file_path: str = None):
    with open(file_path, "rb") as f:
        result = client.files.create(file=f, purpose="assistants")
        return result.id

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
file_id = create_file(client, 'faqs.json')
logging.info(f"Uploaded file: {file_id}")

vectore_store = client.vector_stores.create(name="mechaniGoPH_FAQs")
logging.info(f"Created vector store: {vectore_store.id}")

logging.info("Attaching file to vector store...")
client.vector_stores.files.create(
    vector_store_id=vectore_store.id,
    file_id=file_id
)

logging.info("Saving vector store ID for reusability...")
with open("vector_store_id.json", "w") as f:
    json.dump({"id": vectore_store.id}, f)
"""
Usage:

from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="What time does MechaniGo close?"
    tools=[
        {
            "type": file_search":
            "vector_store_ids": [vector_store.id]
        }
    ]
)
"""
logging.info("Saved!")