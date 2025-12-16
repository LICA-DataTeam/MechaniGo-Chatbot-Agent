# MechaniGo Chatbot Agent (major-refactor-branch)

- Includes components for the chatbot (OpenAI Agents SDK) and the FastAPI endpoint (`/send-message`)

- **Soon**: Streamlit version for a user-friendly interface

## Run locally (`dev`)

- Create a `.env.dev` file and copy the contents of `.env.example` to it, then supply the values:

```bash
cp .env.example > .env.dev
```

- In your terminal session, enter the following command/s:

**bash**:

```bash
export ENVIRONMENT=development
```

***powershell**:

```powershell
$env:ENVIRONMENT="development"
```

- Then run `python main.py`.

## Deploy in prod

- Create a `.env.prod` file and copy the contents of `.env.example` to it. (Same steps as above)

### Configuration

- Settings can be found in `config/settings.py`.

### TODO

- [x] Implement Supabase config (storage)

- [x] `BookingAgent`

    - [x] Fix save functionality for booking ([ADD-233](https://lica-group.atlassian.net/browse/ADD-233))

        - [x] Strengthen `BookingAgent` prompt

- [x] Implement BigQuery config for metrics tracking and analytics

- [x] Optimize response time in `faq_tool` and `MechanicAgent` tool

    - [x] Fix related issue ([ADD-232](https://lica-group.atlassian.net/browse/ADD-232))