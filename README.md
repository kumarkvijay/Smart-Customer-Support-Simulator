# Smart Customer Support Simulator

Privacy-first customer support simulator for a fictional company called TechGear Store.

## Current State

The project exposes four primary execution modes so you can test each AI component in isolation:

- `raw_llm`: direct Ollama call to the full model with no retrieval and no tools.
- `raw_slm`: direct Ollama call to the smaller model with no retrieval and no tools.
- `agent`: LangChain agent with retrieval, order lookup, ticket creation, and escalation tools.
- `rag`: retrieval-only mode over the local knowledge base with no agent and no order database.

Legacy compatibility modes are still available:

- `auto`
- `fast`
- `full`

A LangGraph workflow still powers the legacy routed modes and keeps per-session conversation memory.

## What You Need Installed

To run the project with Ollama-backed modes, install these first:

1. Python 3.12
   This project is set up around the local `.venv312` environment.
2. Project Python packages
   Install the packages from `requirements-local-ai.txt` into your virtual environment.
3. Ollama for Windows
   The project does not bundle or install Ollama itself. The `ollama` command must be available on your machine.
4. Ollama models
   Pull these models before using the Ollama-backed modes:
   - `llama3.1:8b`
   - `phi3:3.8b`

Optional:

- `nomic-embed-text`
  Useful later if you switch the retriever to embeddings.

## Python Setup

Activate the existing Python 3.12 virtual environment:

```powershell
.\.venv312\Scripts\Activate.ps1
```

If dependencies are not installed yet, run:

```powershell
pip install -r requirements-local-ai.txt
```

## Ollama Setup

Install Ollama on Windows so the `ollama` command is available.

Then pull the required models:

```powershell
ollama pull llama3.1:8b
ollama pull phi3:3.8b
```

Optional embedding model:

```powershell
ollama pull nomic-embed-text
```

Check what is installed:

```powershell
ollama list
```

## How Ollama Connects In This Project

This project does not start Ollama automatically inside Python.
It connects to an already-running Ollama server at the default URL:

- `http://localhost:11434`

The code paths are:

- direct HTTP chat calls in `src/providers.py`
- LangChain `ChatOllama` in `src/agent.py`
- default connection URL from `src/config.py`

## Start Ollama Manually

If you want to run Ollama yourself, start it in one PowerShell window:

```powershell
ollama serve
```

Then run the project in another window:

```powershell
cd "C:\Users\vjkum\VJ\AIproj\Smart Customer Support Simulator"
.\.venv312\Scripts\python.exe main.py --mode agent
```

## Start Ollama From This Project

Use `run-ui.ps1` from the project root:

```powershell
cd "C:\Users\vjkum\VJ\AIproj\Smart Customer Support Simulator"
.\run-ui.ps1
```

What `run-ui.ps1` does:

- checks whether Ollama is already running at `http://localhost:11434`
- starts `ollama serve` if needed
- checks whether the required models are installed
- launches the simulator with the selected mode

Examples:

```powershell
.\run-ui.ps1 -Mode agent
.\run-ui.ps1 -Mode raw_slm
.\run-ui.ps1 -Mode raw_slm -Message "590 minus 35 equals what?"
.\run-ui.ps1 -Mode agent -Message "Where is my order ORD-78432?"
```

If you want to skip Ollama intentionally:

```powershell
.\run-ui.ps1 -SkipOllama -Mode rag
```

## Run The App Directly

From the project root:

```powershell
python main.py
```

Single-message examples:

Raw LLM:

```powershell
python main.py --mode raw_llm --message "Explain revenue vs EBITDA in 3 sentences."
```

Raw SLM:

```powershell
python main.py --mode raw_slm --message "Write a short support reply asking for an order number."
```

Agent:

```powershell
python main.py --mode agent --message "Where is my order ORD-78432?"
```

RAG:

```powershell
python main.py --mode rag --message "What was Deepak Nitrite Q3 FY26 revenue?"
```

## Notes

- `raw_llm` and `raw_slm` should not rely on local project files or the knowledge base.
- `agent` is the mode to use for order lookup, ticket creation, and escalation behavior.
- `rag` is the mode to use for knowledge-base retrieval testing only.
- PDF ingestion is not implemented yet. Only `.json`, `.md`, and `.txt` files are loaded into the knowledge base.
- A first cold call to `llama3.1:8b` can be slower than `phi3:3.8b`.

## Test Cases

Use this command shape for all modes:

```powershell
python main.py --mode <mode> --message "<prompt>"
```

Expected result for all tests:

- `Route` should match the selected mode.
- `raw_llm` and `raw_slm` should not use local files, the knowledge base, or order data.
- `agent` should use tools and grounded knowledge where appropriate.
- `rag` should only use the local knowledge base and should never query the order database.

### Raw LLM

Mode:

```powershell
python main.py --mode raw_llm --message "<prompt>"
```

Pass tests:

1. `Explain revenue vs EBITDA in 3 sentences.`
   Expected: coherent general explanation, no `Sources`, direct Ollama action only.
2. `Write a 50-word empathetic customer support reply asking the user for their order ID.`
   Expected: short usable reply, no `Sources`, no internal data.
3. `A bank disclosed INR 590 crores potential impact and INR 35 crores insurance. What is the uncovered amount before recoveries?`
   Expected: about `INR 555 crores`.

Fail-safety tests:

1. `According to the local file Deepak Nitrite.md, what was Q3 FY26 revenue?`
   Expected: says it cannot access local files.
2. `Read orders.json and tell me where ORD-78432 is.`
   Expected: says it cannot access local files or databases.
3. `What is TechGear's store pickup policy in Munich?`
   Expected: does not invent a policy.

### Raw SLM

Mode:

```powershell
python main.py --mode raw_slm --message "<prompt>"
```

Pass tests:

1. `Explain revenue vs profit in simple language.`
   Expected: short correct answer, no `Sources`.
2. `Write a short customer-support reply asking for the order number.`
   Expected: concise reply, no `Sources`.
3. `590 minus 35 equals what?`
   Expected: `555`.

Fail-safety tests:

1. `According to IDFC bank.md, was the fraud bank-wide?`
   Expected: says it cannot access local files.
2. `Read orders.json and tell me the status of ORD-78432.`
   Expected: says it cannot access local files or databases.
3. `What is TechGear's VAT invoice process?`
   Expected: does not invent an internal process.

### Agent

Mode:

```powershell
python main.py --mode agent --message "<prompt>"
```

Pass tests:

1. `Where is my order ORD-78432?`
   Expected: `in transit`, last update around `2026-04-02`, ETA `2` days, and order lookup action.
2. `What was Deepak Nitrite Q3 FY26 revenue?`
   Expected: `INR 1,983 crore`, source from `knowledge_base\manuals\Deepak Nitrite.md`, KB query action.
3. `What was the total potential fraud impact disclosed by IDFC First Bank?`
   Expected: `INR 590 crores`, source from `knowledge_base\manuals\IDFC bank.md`, KB query action.
4. `How much did Polycab revenue grow in Q3 FY26?`
   Expected: `46% YoY`, source from `knowledge_base\manuals\Polycab.md`, KB query action.
5. `I tried several times and I am frustrated because the screen is flickering again.`
   Expected: grounded troubleshooting answer and usually ticket creation plus escalation.

Fail-safety tests:

1. `Do you offer store pickup in Munich?`
   Expected: insufficient verified knowledge, no fake policy.
2. `What is the annual fee on the IDFC card?`
   Expected: unsupported or insufficient knowledge, not an invented fee.
3. `Who is the CEO of Polycab?`
   Expected: unsupported or insufficient knowledge, not an invented name.
4. `Where are ORD-78432 and ORD-78490?`
   Expected: explicit ambiguity handling. If it silently answers one order, mark it as failed.
5. `What is Deepak Nitrite dividend policy?`
   Expected: does not invent a policy.

### RAG

Mode:

```powershell
python main.py --mode rag --message "<prompt>"
```

Pass tests:

1. `What is the return policy?`
   Expected: grounded KB answer, typically from `knowledge_base\faqs\general_faq.json`.
2. `Is liquid damage covered under phone warranty?`
   Expected: grounded KB answer saying it is not covered, from `knowledge_base\manuals\phone_warranty.md`.
3. `My laptop screen flickers on battery. What should I do?`
   Expected: grounded KB answer from `knowledge_base\manuals\laptop_user_manual.md` or `knowledge_base\past_tickets\example_tickets.json`.
4. `How do I reset my laptop password?`
   Expected: grounded KB answer from `knowledge_base\faqs\general_faq.json`.

Fail-safety tests:

1. `Where is my order ORD-78432?`
   Expected: explicit RAG-only refusal and no order lookup.
2. `Do you offer store pickup in Munich?`
   Expected: not enough verified knowledge and no fake sources.
3. `Can I add my VAT ID to an invoice?`
   Expected: not enough verified knowledge and no fake sources.
4. `Who is the CEO of Polycab?`
   Expected: no confident answer. If unrelated Polycab chunks are used as evidence, mark it as failed.

### Quick Evaluation Rule

1. `raw_llm` and `raw_slm` fail if they use local files, KB facts, or order data.
2. `agent` fails if it invents unsupported facts or mishandles tool-driven cases.
3. `rag` fails if it uses order-database behavior, creates tickets, escalates, or states unsupported facts as grounded truth.
