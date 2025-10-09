# MechaniGo-Chatbot-Agent
A prototype AI-assistant for MechaniGo.ph, designed to interact with users and streamline processes within the platform (booking). The assistant leverages the [OpenAI Agents SDK components](https://openai.github.io/openai-agents-python/agents/).

### Features

- Integration with MechaniGo FAQs
- Utilizes OpenAI API for natural language processing
- Powered by OpenAI Agents SDK for agent-based interactions

## Run locally

### Setting Up
Install the required modules and packages:
```bash
pip install -r requirements.txt
```

Make sure to have the following:
- Your OpenAI API key
    - Save it to `.env` or `.streamlit/secrets.toml`
    - **Important**: Store OpenAI API key in `.streamlit/secrets.toml` when you run Streamlit interface
- Google credentials
- The `faqs.json` file (See [File Search](#file-search-vector-store))

### Streamlit

```python
streamlit run app.py
```

### Test scripts

To run the test scripts, use the following format `python -m tests.[name_of_test_script]`

Example:
```bash
python -m tests.test_booking_agent
```

**Important**: Make sure you are in root directory when entering the above command. Otherwise, `cd` into tests directory then run the script directly.

# Components

## Orchestrator Agent (`MechaniGoAgent`)
- The customer facing agent.
- Orchestrates the other sub-agents.

### Sub-Agents (as tools)

#### Bookings Agent
- Handles the scheduling of appointment as well as payment of user.

#### FAQs Agent
- Answers the FAQs. MechaniGo PH FAQ data is indexed as vectore store (See more: [File Search](#file-search-vector-store))

#### Mechanic Agent
- Handles car-related issues.

#### User Info Agent
- Handles the extraction of user information such as name, contact number, address, etc.

## File Search (Vector Store)
The `FAQsAgent` utilizes the [File Search](https://platform.openai.com/docs/guides/tools-file-search) API tools to query MechaniGo PH's FAQ data.

To enable the agent to search the FAQ data, a **vector store ID** must be created and indexed.

### Steps

1. Run `file_search_utils.py` to generate the FAQ data index.
    - This script only needs to be run once (or again when FAQ file changes)
    - To run: `python -m components.utils.file_search_utils`
        - This will create a file named `vector_store_id.json` in the project directory.

# Configuration

If modifications are needed, especially in BigQuery related configurations, update the variables in `config/constants.py`.