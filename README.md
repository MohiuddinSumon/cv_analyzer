# CV Analysis System

A comprehensive system for processing, analyzing, and querying CV/resume documents using OCR and LLMs. Transform your recruitment workflow with AI-powered candidate analysis.

## Features

- **Document Processing**: Handle PDF and Word documents with advanced OCR capabilities
- **Information Extraction**: Extract structured data from CVs using AI models
- **Natural Language Querying**: Ask questions about candidates in plain English
- **Web Interface**: User-friendly Streamlit application with multiple functionality tabs
- **Multi-model Support**: Works with Claude, GPT, and Gemini models
- **Interactive CV Viewing**: View extracted CV information in structured format
- **Conversation Memory**: System remembers context for follow-up questions
- **Profile Management**: Create and manage your own professional profile
- **Job Analysis**: Analyze job descriptions and generate matching scores
- **Cover Letter Generation**: Auto-generate customized cover letters
- **Resume Editor**: Interactive interface for updating resume content

## Use Cases & Benefits

### For Recruiters
- **Time Savings**: Reduce CV review time by up to 75% through automated information extraction
- **Consistent Evaluation**: Standardize candidate information for fair comparison
- **Better Matching**: Quickly identify candidates with specific skills or experience
- **Insights Discovery**: Uncover patterns and connections in candidate data through natural language queries

### For HR Teams
- **Centralized Database**: Keep all candidate information in one organized system
- **Collaborative Hiring**: Share insights with team members through the intuitive interface
- **Reduced Bias**: Focus on skills and qualifications with structured data
- **Efficient Screening**: Pre-screen candidates at scale with intelligent queries

### For Organizations
- **Improved Hiring Efficiency**: Decrease time-to-hire by streamlining CV analysis
- **Better Talent Identification**: Discover ideal candidates that might be overlooked in manual processes
- **Scalable Recruitment**: Process hundreds of CVs quickly during high-volume hiring periods
- **Data-Driven Decisions**: Base hiring decisions on comprehensive candidate analysis

### For Job Seekers
- **Profile Management**: Maintain your professional profile and resume in one place
- **Job Match Analysis**: Get detailed analysis of your fit for specific job postings
- **Automated Cover Letters**: Generate customized cover letters based on job descriptions
- **Resume Enhancement**: Edit and improve your resume with an interactive interface
- **Application History**: Track your job applications and analyses

## Requirements

- Python 3.8+
- Tesseract OCR
- LLM API key (Anthropic Claude, OpenAI GPT, or Google Gemini)
- Poppler (for PDF processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MohiuddinSumon/cv_analyzer.git
cd cv_analyzer
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
   - **Ubuntu**: `sudo apt-get install tesseract-ocr poppler-utils`
   - **macOS**: `brew install tesseract poppler`
   - **Windows**: 
     - Download Tesseract from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
     - Add Tesseract to your PATH environment variable
     - Download Poppler binaries for Windows

5. Create a `.env` file with your API keys (choose one):
```
ANTHROPIC_API_KEY=your_anthropic_api_key
# or
OPENAI_API_KEY=your_openai_api_key
# or
GOOGLE_API_KEY=your_google_api_key
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

6. Use the various tabs for different functionalities:

### Tab Features

#### Chat Assistant
- Query CV information using natural language
- View conversation history
- Get insights about candidates

#### CV Details
- View structured CV information
- Access detailed candidate profiles
- Review extracted data in JSON format

#### Profile
- Upload and manage your personal resume
- Add professional links (Portfolio, GitHub, LinkedIn)
- Include additional information
- View analyzed profile information

#### Job Analysis
- Input job descriptions manually or via URL
- Get detailed job match analysis
- Generate customized cover letters
- Track application history
- Download analyses and cover letters
- Compare your profile against job requirements

#### Resume Editor
- Edit personal information interactively
- Update skills and experience
- Modify work history details
- Save changes in real-time

7. View detailed CV information by clicking on a candidate name in the sidebar

8. When you're done, deactivate the virtual environment:
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
├── data
│    └── sample_cvs/             # Sample CV files for testing
│        ├── candidate1.pdf
│        └── candidate2.docx
└── README.md               # This file
```

## How It Works

1. **Document Processing**: 
   - Extracts text from PDFs and Word documents
   - Uses OCR for scanned documents or images with adaptive thresholding
   - Handles complex document layouts and formatting

2. **Information Extraction**:
   - Sends CV text to LLM (Claude, GPT, or Gemini)
   - Uses prompt engineering to extract structured information
   - Organizes data into categories:
     - Personal information (name, contact details)
     - Education history
     - Work experience
     - Skills (technical, soft, languages)
     - Projects
     - Certifications

3. **Data Storage**:
   - Stores structured CV data in a JSON database
   - Maintains efficient indexing for quick retrieval
   - Persists data between application sessions

4. **Query Interface**:
   - Uses LLM to interpret natural language queries about candidates
   - Maintains conversation context for follow-up questions
   - Provides informative responses based on candidate data
   - Supports comparison of multiple candidates

5. **CV Viewer**:
   - Displays structured CV information in an easy-to-read format
   - Allows quick navigation between different candidates
   - Presents complete candidate profiles

## Technical Details

The system uses a modular architecture with several key components:

- **CVProcessor**: Handles document processing with fallback methods for difficult documents
- **CVAnalyzer**: Extracts structured information using LLM models
- **CVDatabase**: Manages data storage and retrieval
- **CVQueryEngine**: Processes natural language queries about candidates
- **Streamlit Interface**: Provides a user-friendly web interface

## Customization

- **OCR Engine**: You can modify the OCR engine in the `CVProcessor` class
- **LLM Provider**: Choose between Anthropic Claude, OpenAI GPT, or Google Gemini
- **Database Storage**: Change the storage format in the `CVDatabase` class
- **UI Customization**: Extend the Streamlit interface to add custom features

## Performance Considerations

- OCR processing is resource-intensive for large documents
- LLM API calls incur costs based on your provider's pricing
- Processing speed depends on document complexity and selected models

## Future Enhancements

- CV similarity scoring
- Candidate recommendation engine
- Skills gap analysis
- Integration with applicant tracking systems
- Export to common HR formats

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.