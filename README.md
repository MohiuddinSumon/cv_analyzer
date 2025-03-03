# CV Analysis System

A comprehensive system for processing, analyzing, and querying CV/resume documents using OCR and LLMs.

## Features

- **Document Processing**: Handle PDF and Word documents with OCR capabilities
- **Information Extraction**: Extract structured data from CVs using AI
- **Natural Language Querying**: Ask questions about candidates in plain English
- **Web Interface**: User-friendly Streamlit application for uploading and analyzing CVs
- **Multi-model Support**: Works with Claude or GPT models

## Requirements

- Python 3.8+
- Tesseract OCR
- LLM API key (Anthropic Claude or OpenAI GPT)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cv-analysis-system.git
cd cv-analysis-system
```

2. Create and activate a virtual environment:

   **Using venv (Python's built-in module)**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

   **Using virtualenv (alternative)**
   ```bash
   # Install virtualenv if not already installed
   pip install virtualenv
   
   # Create virtual environment
   virtualenv .venv
   
   # Activate virtual environment
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

   Your command prompt should now show the virtual environment name, indicating it's active.

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
# or
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Ensure your virtual environment is activated (you should see `.venv` in your command prompt)

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the provided URL (typically http://localhost:8501)

4. Upload CV files through the sidebar

5. Process the CVs using the "Process CVs" button

6. Use the chat interface to query information about candidates:
   - "Find candidates with Python experience"
   - "Who has a Master's degree in Computer Science?"
   - "Compare John and Sarah's skills"
   - "Which candidate has the most experience in machine learning?"

7. When you're done, deactivate the virtual environment:
```bash
deactivate
```

## Project Structure

```
cv-analysis-system/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── .venv/                  # Virtual environment directory
├── cv_database.json        # Database file for stored CV data
├── sample_cvs/             # Sample CV files for testing
│   ├── candidate1.pdf
│   └── candidate2.docx
└── README.md               # This file
```

## How It Works

1. **Document Processing**: 
   - Extracts text from PDFs and Word documents
   - Uses OCR for scanned documents or images
   - Preprocesses images for better OCR results

2. **Information Extraction**:
   - Sends CV text to LLM (Claude or GPT)
   - Uses prompt engineering to extract structured information
   - Organizes data into categories (personal info, education, experience, skills, etc.)

3. **Data Storage**:
   - Stores structured CV data in a JSON database
   - Allows for efficient retrieval and querying

4. **Query Interface**:
   - Uses LLM to interpret natural language queries
   - Maintains conversation context for follow-up questions
   - Returns relevant information about candidates

## Customization

- **OCR Engine**: You can modify the OCR engine in the `CVProcessor` class
- **LLM Provider**: Choose between Anthropic Claude and OpenAI GPT in the `CVAnalyzer` class
- **Database Storage**: Change the storage format in the `CVDatabase` class

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.