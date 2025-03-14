# SupportBotAgent

## Overview

SupportBotAgent is an AI-powered support bot that processes PDF documents, stores relevant content in a vector database, and provides intelligent responses to user queries. It utilizes OpenAI's GPT-4-turbo model to generate responses and CrewAI for response validation.

## Features

* Extracts text from PDF documents.
* Stores extracted content in a vector database for efficient retrieval.
* Uses OpenAI's GPT-4-turbo for answering user queries.
* Evaluates responses for relevance and accuracy using CrewAI.
* Adjusts responses dynamically based on feedback.
* Logs all interactions and processes for auditing and debugging.


## Technologies Used

* Python
* FAISS (Facebook AI Similarity Search)
* OpenAI GPT-4 Turbo
* CrewAI
* dotenv (for managing environment variables)

## Prerequisites

* Python 3.8+
* OpenAI API Key (stored in a `.env` file)
* Required Python dependencies (install via `requirements.txt`)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/support-bot.git
   cd support-bot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up the `.env` file:
   ```sh
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. Place your PDF document in the `dataset/` directory.

## Usage

Run the script by executing:

```sh
python support_bot.py
```

The script will:

1. Load the API key.
2. Extract and store PDF content in a vector database.
3. Wait for user queries and provide intelligent responses.
4. Evaluate and adjust responses dynamically.

## Logging

All interactions, queries, responses, and feedback are logged in `logs.txt` for auditing and debugging.

## Example Query

```
Enter your query: What is the main topic of the document?
Final Response: The document discusses XYZ in detail, covering aspects A, B, and C.
```

## License

## Future Enhancements

* **Support for Multiple PDFs:** Enable querying across multiple documents.
* **Improved Feedback Mechanism:** Enhance feedback collection for better response refinement.
* **Integration with Other LLMs:** Extend support for additional models beyond OpenAI.
* **Better Retrieval Mechanism:** Improve vector search for more precise document retrieval.
* **Advanced Preprocessing Techniques:** Implement better text cleaning and chunking methods.
* **Additional AI Agents:** Introduce more specialized agents for tasks like summarization and document classification.
