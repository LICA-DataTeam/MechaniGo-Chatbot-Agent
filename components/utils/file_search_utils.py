# Documentation/Guide: https://platform.openai.com/docs/guides/tools-file-search
from dotenv import load_dotenv
from openai import OpenAI
import logging
import json
import os

# IMPORTANT:
# - This script only needs to be run once (or again when their file changes)
# - Once saved, the FAQ data and MechaniGo PH CMS blog posts are indexed and OpenAI can query it now

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

def create_vector_store(client: OpenAI, filename: str, vector_store_name: str, save: bool = False):
    """
    Helper function for creating vector store IDs for MechaniGo chatbot agent.
    Once you created your artifacts, use OpenAI's `file_search` module:

    ```python
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model="gpt-5,
        input="What time does MechaniGo close?",
        tools=[
            {"type": "file_search", "vector_store_ids": [vector_store.id]}
        ]
    )
    ```

    Args:
        client (`OpenAI`): OpenAI client (**Important**: API key)
        filename (`str`): Your vector store's artifact file name. This will be saved in the root directory.
        vector_store_name (`str`): Your vector store's name.
        save (`bool`): Whether or not you want to save the vector artifact.
    """
    file_id = create_file(client, filename)
    logging.info(f"Uploaded file: {file_id}")

    vector_store = client.vector_stores.create(name=vector_store_name)
    logging.info(f"Created vector store: {vector_store.id}")
    logging.info(f"Attaching files to vector store...")
    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_id
    )

    if save:
        save

def save_vector_artifact(filename: str, vector_store):
    with open(filename, "w") as f:
        json.dump({"id": vector_store.id}, f)
        logging.info("Saved.")

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    current_dir = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    files = ["faqs.json", "blog-posts.json"] # You can add more here, then add another elif block below
    
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.exists(file_path):
            logging.info(f"Exists: {file}")
            if file == "faqs.json":
                create_vector_store(client, file, "mechaniGoPH_FAQs", save=True)
                logging.info(f"Vector store for {file} indexed and created.")
            elif file == "blog-posts.json":
                create_vector_store(client, file, "mechaniGoPH_blogposts", save=True)
                logging.info(f"Vector store for {file} indexed and created.")
        else:
            logging.info(f"{file} does not exist.")

main()