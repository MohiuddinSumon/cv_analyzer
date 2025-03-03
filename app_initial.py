import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

# LLM integration
import anthropic
import cv2
import docx
import numpy as np
import openai

# Document processing libraries
import PyPDF2
import pytesseract

# Web interface
import streamlit as st
from pdf2image import convert_from_path
from streamlit_chat import message
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv


class CVProcessor:
    """Handles the document processing and information extraction from CVs."""

    def __init__(self, ocr_engine="tesseract"):
        """Initialize the CV processor with the specified OCR engine."""
        self.ocr_engine = ocr_engine
        # Configure OCR
        if ocr_engine == "tesseract":
            pytesseract.pytesseract.tesseract_cmd = (
                r"/usr/bin/tesseract"  # Update this path as needed
            )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files using OCR if needed."""
        try:
            # First try direct text extraction
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"

                # If direct extraction yields minimal text, use OCR
                if len(text.strip()) < 100:
                    logger.info(
                        f"Using OCR for {file_path} as direct extraction yielded minimal text"
                    )
                    return self._extract_text_with_ocr(file_path)
                return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            # Fallback to OCR
            return self._extract_text_with_ocr(file_path)

    def _extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text from PDF using OCR."""
        try:
            # Convert PDF to images
            images = convert_from_path(file_path)
            text = ""

            for i, image in enumerate(images):
                # Convert PIL image to OpenCV format
                open_cv_image = np.array(image)
                open_cv_image = open_cv_image[
                    :, :, ::-1
                ].copy()  # RGB to BGR conversion

                # Preprocess image for better OCR results
                gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]

                # Perform OCR
                page_text = pytesseract.image_to_string(thresh)
                text += page_text + "\n"

            return text
        except Exception as e:
            logger.error(f"OCR processing failed for {file_path}: {str(e)}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return ""

    def process_document(self, file_path: str) -> str:
        """Process a document and extract its text content."""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_extension in [".docx", ".doc"]:
            return self.extract_text_from_docx(file_path)
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return ""


class CVAnalyzer:
    """Uses LLM to extract structured information from CV text."""

    def __init__(self, llm_provider="anthropic", api_key=None):
        """Initialize the CV analyzer with the specified LLM provider."""
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = "claude-3-7-sonnet-20250219"
        elif llm_provider == "openai":
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4"
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def extract_cv_information(self, cv_text: str) -> Dict[str, Any]:
        """Extract structured information from CV text using LLM."""
        prompt = f"""
        You are a CV analysis expert. Extract the following structured information from the CV text below:
        
        1. Personal Information (name, email, phone, location)
        2. Education History (institution, degree, field, graduation_date)
        3. Work Experience (company, role, duration, responsibilities, achievements)
        4. Skills (technical, soft, languages)
        5. Projects (name, description, technologies)
        6. Certifications (name, issuer, date)
        
        Return the information in JSON format with these exact categories. If any information is not found, include the category with an empty value or array.
        
        CV Text:
        {cv_text}
        """

        try:
            if self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.content[0].text
            elif self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a CV analysis expert."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                )
                result = response.choices[0].message.content

            # Extract JSON from the response
            json_match = re.search(r"```json\n([\s\S]*?)\n```", result)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = result

            # Clean up any non-JSON content
            json_str = re.sub(r"^[^{]*", "", json_str)
            json_str = re.sub(r"[^}]*$", "", json_str)

            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error extracting CV information: {str(e)}")
            # Return a basic empty structure on error
            return {
                "personal_information": {},
                "education_history": [],
                "work_experience": [],
                "skills": {},
                "projects": [],
                "certifications": [],
            }


class CVDatabase:
    """Stores and manages extracted CV information."""

    def __init__(self, storage_path="cv_database.json"):
        """Initialize the CV database with the specified storage path."""
        self.storage_path = storage_path
        self.cvs = {}
        self.load_database()

    def load_database(self):
        """Load the CV database from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as file:
                    self.cvs = json.load(file)
        except Exception as e:
            logger.error(f"Error loading CV database: {str(e)}")
            self.cvs = {}

    def save_database(self):
        """Save the CV database to storage."""
        try:
            with open(self.storage_path, "w") as file:
                json.dump(self.cvs, file, indent=2)
        except Exception as e:
            logger.error(f"Error saving CV database: {str(e)}")

    def add_cv(self, cv_id: str, cv_data: Dict[str, Any]):
        """Add a CV to the database."""
        self.cvs[cv_id] = cv_data
        self.save_database()

    def get_cv(self, cv_id: str) -> Optional[Dict[str, Any]]:
        """Get a CV from the database by ID."""
        return self.cvs.get(cv_id)

    def get_all_cvs(self) -> Dict[str, Dict[str, Any]]:
        """Get all CVs from the database."""
        return self.cvs

    def search_cvs(self, query: str) -> List[str]:
        """Basic search functionality for CVs."""
        query = query.lower()
        results = []

        for cv_id, cv_data in self.cvs.items():
            # Search in skills
            skills = cv_data.get("skills", {})
            all_skills = []
            for skill_category in skills.values():
                if isinstance(skill_category, list):
                    all_skills.extend(skill_category)

            if any(query in skill.lower() for skill in all_skills):
                results.append(cv_id)
                continue

            # Search in work experience
            work_exp = cv_data.get("work_experience", [])
            for exp in work_exp:
                if (
                    query in exp.get("company", "").lower()
                    or query in exp.get("role", "").lower()
                ):
                    results.append(cv_id)
                    break

            # Search in education
            education = cv_data.get("education_history", [])
            for edu in education:
                if (
                    query in edu.get("institution", "").lower()
                    or query in edu.get("field", "").lower()
                ):
                    results.append(cv_id)
                    break

        return results


class CVQueryEngine:
    """Handles natural language queries about the CV database."""

    def __init__(self, cv_database: CVDatabase, llm_provider="anthropic", api_key=None):
        """Initialize the query engine with the CV database and LLM provider."""
        self.cv_database = cv_database
        self.llm_provider = llm_provider
        self.conversation_context = []

        if llm_provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = "claude-3-7-sonnet-20250219"
        elif llm_provider == "openai":
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4"
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def add_to_context(self, role: str, content: str):
        """Add a message to the conversation context."""
        self.conversation_context.append({"role": role, "content": content})
        # Keep only the last 10 messages for context window management
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def process_query(self, query: str) -> str:
        """Process a natural language query about the CVs."""
        all_cvs = self.cv_database.get_all_cvs()

        # Create a compact representation of CVs for context
        cv_summary = {}
        for cv_id, cv_data in all_cvs.items():
            name = cv_data.get("personal_information", {}).get(
                "name", f"Candidate {cv_id}"
            )

            # Extract key skills
            skills = []
            for skill_category, skill_list in cv_data.get("skills", {}).items():
                if isinstance(skill_list, list):
                    skills.extend(skill_list[:5])  # Limit to top 5 skills per category

            # Extract latest work experience
            latest_work = "No work experience"
            work_exp = cv_data.get("work_experience", [])
            if work_exp:
                latest = work_exp[0]
                latest_work = f"{latest.get('role', 'Role')} at {latest.get('company', 'Company')}"

            # Extract latest education
            latest_edu = "No education information"
            education = cv_data.get("education_history", [])
            if education:
                latest = education[0]
                latest_edu = f"{latest.get('degree', 'Degree')} in {latest.get('field', 'Field')} from {latest.get('institution', 'Institution')}"

            cv_summary[cv_id] = {
                "name": name,
                "latest_work": latest_work,
                "latest_education": latest_edu,
                "top_skills": skills[:10],  # Limit to top 10 skills overall
            }

        # Prepare the conversation for the LLM
        system_prompt = f"""
        You are a CV analysis assistant. You have access to {len(all_cvs)} CV profiles with the following summary information:
        
        {json.dumps(cv_summary, indent=2)}
        
        Provide helpful, accurate responses to queries about these candidates. If you need more specific information that's not in the summary, indicate that you may need to look into the detailed CV.
        """

        # Add the query to the context
        self.add_to_context("user", query)

        try:
            if self.llm_provider == "anthropic":
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.conversation_context)

                response = self.client.messages.create(
                    model=self.model, max_tokens=2000, messages=messages
                )
                result = response.content[0].text
            elif self.llm_provider == "openai":
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.conversation_context)

                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=2000
                )
                result = response.choices[0].message.content

            # Add the response to the context
            self.add_to_context("assistant", result)
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your query. Please try again or rephrase your question. Error details: {str(e)}"

    def get_detailed_cv_info(self, cv_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific CV."""
        return self.cv_database.get_cv(cv_id)


class CVAnalysisSystem:
    """Main class that coordinates the CV analysis system components."""

    def __init__(
        self,
        ocr_engine="tesseract",
        llm_provider="anthropic",
        api_key=None,
        storage_path="cv_database.json",
    ):
        """Initialize the CV analysis system."""
        self.cv_processor = CVProcessor(ocr_engine=ocr_engine)
        self.cv_analyzer = CVAnalyzer(llm_provider=llm_provider, api_key=api_key)
        self.cv_database = CVDatabase(storage_path=storage_path)
        self.query_engine = CVQueryEngine(
            self.cv_database, llm_provider=llm_provider, api_key=api_key
        )

    def process_cv_file(self, file_path: str) -> str:
        """Process a CV file and store its structured information."""
        try:
            # Generate a unique ID for the CV
            cv_id = os.path.basename(file_path).split(".")[0]

            # Check if this CV has already been processed
            if self.cv_database.get_cv(cv_id):
                logger.info(f"CV {cv_id} already exists in the database")
                return cv_id

            # Extract text from the document
            logger.info(f"Extracting text from {file_path}")
            cv_text = self.cv_processor.process_document(file_path)

            if not cv_text:
                logger.error(f"Failed to extract text from {file_path}")
                return None

            # Extract structured information using LLM
            logger.info(f"Analyzing CV content for {file_path}")
            cv_data = self.cv_analyzer.extract_cv_information(cv_text)

            # Store the CV data
            logger.info(f"Storing CV data for {file_path}")
            self.cv_database.add_cv(cv_id, cv_data)

            return cv_id
        except Exception as e:
            logger.error(f"Error processing CV file {file_path}: {str(e)}")
            return None

    def process_cv_directory(self, directory_path: str) -> List[str]:
        """Process all CV files in a directory."""
        processed_ids = []

        for filename in os.listdir(directory_path):
            if filename.endswith((".pdf", ".docx", ".doc")):
                file_path = os.path.join(directory_path, filename)
                cv_id = self.process_cv_file(file_path)
                if cv_id:
                    processed_ids.append(cv_id)

        return processed_ids

    def query_cvs(self, query: str) -> str:
        """Query the CV database using natural language."""
        return self.query_engine.process_query(query)

    def get_cv_details(self, cv_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific CV."""
        return self.cv_database.get_cv(cv_id)


# Streamlit web interface
def run_web_interface():
    """Run the Streamlit web interface for the CV analysis system."""
    st.set_page_config(page_title="CV Analysis System", layout="wide")

    # Initialize the CV analysis system
    if "cv_system" not in st.session_state:
        st.session_state.cv_system = CVAnalysisSystem(
            llm_provider="anthropic", api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("CV Analysis System")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("Upload and Process CVs")

        uploaded_files = st.file_uploader(
            "Upload CV files", type=["pdf", "docx", "doc"], accept_multiple_files=True
        )

        if uploaded_files:
            process_button = st.button("Process CVs")

            if process_button:
                with st.spinner("Processing CV files..."):
                    for uploaded_file in uploaded_files:
                        # Save the uploaded file temporarily
                        temp_file_path = f"temp_{uploaded_file.name}"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Process the CV
                        cv_id = st.session_state.cv_system.process_cv_file(
                            temp_file_path
                        )

                        if cv_id:
                            st.success(f"Processed CV: {uploaded_file.name}")
                        else:
                            st.error(f"Failed to process CV: {uploaded_file.name}")

                        # Remove the temporary file
                        os.remove(temp_file_path)

        # Display the list of processed CVs
        st.header("Processed CVs")
        all_cvs = st.session_state.cv_system.cv_database.get_all_cvs()

        if all_cvs:
            for cv_id, cv_data in all_cvs.items():
                name = cv_data.get("personal_information", {}).get(
                    "name", f"Candidate {cv_id}"
                )
                st.write(f"- {name} ({cv_id})")
        else:
            st.write("No CVs processed yet")

    # Main area for chat interface
    st.header("CV Query Assistant")

    # Display chat history
    for i, (query, response) in enumerate(st.session_state.chat_history):
        message(query, is_user=True, key=f"user_{i}")
        message(response, key=f"assistant_{i}")

    # Chat input
    user_query = st.text_input("Ask a question about the CVs:", key="user_query")

    if st.button("Send") or (
        user_query and user_query != st.session_state.get("last_query", "")
    ):
        if user_query:
            # Save the query to prevent duplicate processing on rerun
            st.session_state.last_query = user_query

            with st.spinner("Processing your query..."):
                response = st.session_state.cv_system.query_cvs(user_query)
                st.session_state.chat_history.append((user_query, response))

            # Clear the input after sending
            st.experimental_rerun()


if __name__ == "__main__":

    load_dotenv()

    # Example usage
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set the ANTHROPIC_API_KEY environment variable")
        exit(1)

    cv_system = CVAnalysisSystem(llm_provider="anthropic", api_key=api_key)

    # Process a directory of CVs
    cv_dir = "data/sample_cvs"
    if os.path.exists(cv_dir):
        print(f"Processing CVs in {cv_dir}...")
        processed_ids = cv_system.process_cv_directory(cv_dir)
        print(f"Processed {len(processed_ids)} CVs")

    # Run the web interface
    run_web_interface()
