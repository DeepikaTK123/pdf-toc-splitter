import os
import re
import argparse
import time
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from io import BytesIO

# Flask imports
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_file

# PDF processing imports
import google.generativeai as genai
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
from google.ai import generativelanguage as glm
from google.ai.generativelanguage import Part
from google.generativeai.types import GenerationConfig
from google.generativeai import configure, GenerativeModel
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define directory permission check function
def check_directory_permissions(directory_path):
    """
    Check if a directory exists and has the right permissions,
    returning a detailed report.
    """
    if not os.path.exists(directory_path):
        return {
            "exists": False,
            "is_dir": False,
            "readable": False,
            "writable": False,
            "error": f"Path {directory_path} does not exist"
        }
        
    is_dir = os.path.isdir(directory_path)
    readable = os.access(directory_path, os.R_OK) if is_dir else False
    writable = os.access(directory_path, os.W_OK) if is_dir else False
    
    result = {
        "exists": True,
        "is_dir": is_dir,
        "readable": readable,
        "writable": writable,
        "error": None
    }
    
    if not is_dir:
        result["error"] = f"Path {directory_path} is not a directory"
    elif not readable:
        result["error"] = f"Directory {directory_path} is not readable"
    elif not writable:
        result["error"] = f"Directory {directory_path} is not writable"
    
    # Try to create a test file
    if is_dir and readable and writable:
        test_file = os.path.join(directory_path, ".permission_test")
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            result["test_write"] = True
        except Exception as e:
            result["test_write"] = False
            result["error"] = f"Failed to write test file: {str(e)}"
    
    return result

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Get absolute paths for all directories
base_dir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['AI_PDF_DATA'] = os.path.join(base_dir, 'AI-PDF-Data')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Configure logging to also output to console
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more details

# Create console handler with a higher log level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Ensure upload directories exist with proper permissions
logger.info("Checking application directories...")
directories = {
    "upload": app.config['UPLOAD_FOLDER'],
    "ai_pdf_data": app.config['AI_PDF_DATA']
}

for name, directory in directories.items():
    result = check_directory_permissions(directory)
    
    if not result["exists"]:
        logger.warning(f"{name} directory does not exist at {directory}. Attempting to create...")
        try:
            os.makedirs(directory, mode=0o755, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            # Check again after creation
            result = check_directory_permissions(directory)
        except Exception as e:
            logger.error(f"Failed to create {name} directory: {str(e)}")
    
    if result["error"]:
        logger.error(f"Directory {name} check failed: {result['error']}")
    else:
        logger.info(f"Directory {name} at {directory} is ready for use")
        
# Log system info
logger.info(f"Running as user: {os.getuid()}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

# Configuration - copied from split_pdf.py
IMAGE_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], "input_images")
OCR_OUTPUT_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], "ocr_output")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get API key from environment variable
MODEL_NAME = "gemini-2.0-flash"  # Using Gemini Flash model for OCR capabilities
DELAY_BETWEEN_REQUESTS = 2  # Seconds to wait between API calls
MAX_PAGES_TO_CHECK = 20
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]  # Supported image formats

# Initialize Gemini API if key is provided
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.warning("GEMINI_API_KEY environment variable not set. OCR functionality will not work.")

# Function imports from split_pdf.py
def create_unique_folders(input_pdf_path, file_directory="file_directory"):
    """
    Create unique folders for the current execution based on input filename and timestamp.
    Args:
        input_pdf_path (str): Path to the input PDF file
        file_directory (str): Directory to contain all created folders
    Returns:
        dict: Dictionary containing paths to the created folders
    """
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_pdf_path))[0]
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create unique folder name
    unique_id = f"{base_filename}_{timestamp}"
    
    # Make sure the file_directory exists
    os.makedirs(file_directory, exist_ok=True)
    
    # Create folder structure inside file_directory
    folders = {
        "root": os.path.join(file_directory, unique_id),
        "images": os.path.join(file_directory, unique_id, "input_images"),
        "split_pdf": os.path.join(file_directory, unique_id, "splitted_pdfs"),
        "ocr_output": os.path.join(file_directory, unique_id, "ocr_output")
    }
    
    # Create directories
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Created folder: {folder}")
        
    return folders

def setup_gemini(api_key):
    """Initialize Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    # Use Gemini model that supports vision capabilities
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

def split_pdf(input_path, output_directory=None, ranges=None, individual_pages=False):
    """
    Split a PDF file into multiple PDF files based on page ranges or as individual pages.
    
    Args:
        input_path (str): Path to the input PDF file
        output_directory (str, optional): Directory to save the output files. Defaults to same directory as input.
        ranges (list, optional): List of page ranges to split into. Format: [(start1, end1), (start2, end2), ...]
        individual_pages (bool, optional): If True, split into individual pages. Defaults to False.
    
    Returns:
        list: List of paths to the created PDF files
    """
    # Validate input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Set output directory
    if output_directory is None:
        output_directory = os.path.dirname(input_path) or "."
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Open the PDF file
    pdf = PdfReader(input_path)
    total_pages = len(pdf.pages)
    
    created_files = []
    
    # Split by individual pages
    if individual_pages:
        for i in range(total_pages):
            output = PdfWriter()
            output.add_page(pdf.pages[i])
            
            output_filename = f"{base_filename}_page_{i+1}.pdf"
            output_path = os.path.join(output_directory, output_filename)
            
            with open(output_path, "wb") as output_file:
                output.write(output_file)
            
            created_files.append(output_path)
    
    # Split by ranges
    elif ranges:
        for i, page_range in enumerate(ranges):
            start, end = page_range
            
            # Adjust for 0-based indexing
            start_idx = start - 1
            end_idx = min(end, total_pages)
            
            # Validate range
            if start_idx < 0 or start_idx >= total_pages:
                logger.warning(f"Invalid start page {start}. Skipping range {start}-{end}.")
                continue
            
            output = PdfWriter()
            for j in range(start_idx, end_idx):
                output.add_page(pdf.pages[j])
            
            output_filename = f"{base_filename}_pages_{start}-{end}.pdf"
            output_path = os.path.join(output_directory, output_filename)
            
            with open(output_path, "wb") as output_file:
                output.write(output_file)
            
            created_files.append(output_path)
    
    return created_files

def parse_range(range_str):
    """Parse a range string like '1-5' into a tuple (1, 5)"""
    try:
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            return (start, end)
        else:
            # Single page treated as a range of one page
            page = int(range_str)
            return (page, page)
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}")

def convert_pdf_to_images(pdf_path, output_dir=None, dpi=300, image_format="png", first_page=None, last_page=None):
    """
    Convert each page of a PDF to an individual image file.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_dir (str, optional): Directory to save the output images. Defaults to a folder with the PDF name.
        dpi (int, optional): DPI (dots per inch) for the output images. Higher values mean better quality but larger files.
        image_format (str, optional): Format for the output images (png, jpg, tiff). Defaults to "png".
        first_page (int, optional): First page to convert (1-based). Defaults to None (start from first page).
        last_page (int, optional): Last page to convert (1-based). Defaults to None (end at last page).
    
    Returns:
        list: List of paths to the created image files
    """
    # Validate input file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input file not found: {pdf_path}")
    
    # Set output directory
    if output_dir is None:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = f"{pdf_name}_images"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert PDF pages to images
    logger.info(f"Converting PDF to images with DPI={dpi}...")
    images = convert_from_path(
        pdf_path, 
        dpi=dpi,
        first_page=first_page,
        last_page=last_page
    )
    
    # Save each image
    image_paths = []
    total_images = len(images)
    
    for i, image in enumerate(images):
        # Calculate page number (1-based)
        page_num = i + 1
        if first_page:
            page_num = first_page + i
        
        # Create filename
        image_path = os.path.join(output_dir, f"page_{page_num}.{image_format}")
        
        # Save the image
        image.save(image_path, format=image_format.upper())
        image_paths.append(image_path)
    
    logger.info(f"Successfully converted {total_images} pages to {image_format.upper()} images")
    logger.info(f"Images saved to: {os.path.abspath(output_dir)}")
    
    return image_paths

def is_table_of_contents(text):
    """
    Determine if the text represents a table of contents page.
    
    This function first checks if Gemini has explicitly responded with 'YES' in the first line,
    indicating this is a table of contents page. If not, it falls back to pattern matching.
    
    Pattern matching looks for:
    - The phrase "table of contents" or "contents" as a heading
    - A structured list of chapters/sections with page numbers
    - Page number patterns typically found in a TOC
    - Explicit confirmation from Gemini that this is a TOC
    """
    # Check if the first line contains a clear YES response
    first_line = text.strip().split('\n')[0].strip().upper()
    if first_line == "YES":
        return True
    if first_line == "NO":
        return False
    
    # If no clear YES/NO, fall back to pattern matching
    # Check for explicit confirmation from Gemini
    confirmation_patterns = [
        r'yes,?\s+this\s+(page\s+)?is\s+(a\s+)?table\s+of\s+contents',
        r'this\s+(page\s+)?is\s+(a\s+)?table\s+of\s+contents',
        r'table\s+of\s+contents\s+(page|detected)',
        r'this\s+(is\s+)?(the\s+)?table\s+of\s+contents'
    ]
    
    for pattern in confirmation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Check for TOC title variations
    toc_patterns = [
        r'\btable\s+of\s+contents\b',
        r'^\s*contents\s*$',
        r'^\s*table\s+of\s+contents\s*$'
    ]
    
    for pattern in toc_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True 
    
    # Check for typical TOC structure: chapter/section titles followed by page numbers
    # This looks for multiple lines with text followed by numbers or dots and numbers
    toc_structure = re.findall(r'.*?\.{2,}|\d+\s*$|.*?\s+\d+\s*$', text, re.MULTILINE)
    if len(toc_structure) >= 3:  # At least 3 entries that match this pattern
        return True
    
    # Check for consecutive numbered items which might indicate a TOC
    numbered_items = re.findall(r'^\s*\d+\..*$', text, re.MULTILINE)
    if len(numbered_items) >= 3:
        return True
    
    return False

def process_images_toc(folder_path, output_folder, pdfs_folder, model, max_pages_to_check=MAX_PAGES_TO_CHECK):
    """
    Process images to identify table of contents pages.

    Args:
        folder_path (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where OCR output is saved.
        pdfs_folder (str): Path to the folder where splitted pdf pages saved
        model: The Gemini model.
        max_pages_to_check (int): Maximum number of pages to check for TOC.

    Returns:
        tuple: A tuple containing:
            - A list of dictionaries, where each dictionary contains the OCR text and page number of a table of contents page.
            - The PDF page number where the table of contents ends.
            - A list of PDF file paths containing TOC pages.
            - Error message if TOC not found in max_pages_to_check.
    """

    toc_page_list = []
    toc_pages = []
    in_toc_section = False
    toc_start_page = None
    toc_end_page = None
    error_message = None
    # Use the specified prompt template or the default
    prompt_template = """TASK: Table of Contents Detection
                    SYSTEM OBJECTIVE: Analyze PDF pages to identify table of contents (TOC) pages.

                    INPUT: A single PDF page image
                    OUTPUT FORMAT:
                    - FIRST LINE MUST BE ONLY "YES" or "NO" (capitalized)
                    - Following lines should contain your reasoning

                    DETECTION CRITERIA:
                    •⁠  ⁠Heading text containing "Contents," "Table of Contents," "Index," or similar terminology
                    •⁠  ⁠Presence of hierarchical listing of document sections/chapters
                    •⁠  ⁠Alignment of page numbers (typically right-aligned)
                    •⁠  ⁠Indentation patterns indicating hierarchy
                    •⁠  ⁠Position early in document sequence
                    •⁠  ⁠Formatting variations (bold/italic/font size differences) to indicate levels

                    REASONING PROCESS: First identify visual patterns matching TOC structures, then confirm by checking for multiple criteria above.

                    IMPORTANCE: High precision required. Only classify definitive TOC pages as "YES".

                    REMEMBER: Your first line MUST be ONLY "YES" or "NO" with no additional text."""

    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    #  Count total number of image files with supported formats
    total_images = len([
        f for f in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, f)) and 
        Path(f).suffix.lower() in supported_formats
    ])
    
    if total_images == 0:
        logging.error(f"No supported image files found in {folder_path}")
        return [], None, [], "No supported image files found"
    
    # Limit pages to check to either max_pages_to_check or total_images, whichever is smaller
    pages_to_check = min(max_pages_to_check, total_images)
    logger.info(f"Found {total_images} images, checking up to {pages_to_check} pages for ToC detection")

    for page_num in range(1, pages_to_check + 1):
        logger.info(f"Processing page {page_num}/{pages_to_check} for TOC detection...")
        image_file = f"page_{page_num}.png"
        image_path = os.path.join(folder_path, image_file)

        if not os.path.exists(image_path):
            found = False
            for ext in supported_formats:
                alt_file = f"page_{page_num}{ext}"
                alt_path = os.path.join(folder_path, alt_file)
                if os.path.exists(alt_path):
                    image_file = alt_file
                    image_path = alt_path
                    found = True
                    break

            if not found:
                logger.warning(f"Could not find image file for page {page_num}, skipping")
                continue

        try:
            logger.info(f"Loading image: {image_file}")
            img = Image.open(image_path)
            logger.info("Sending image to Gemini for analysis...")

            response = model.generate_content([prompt_template, img])
            extracted_text = response.text
            logger.info("Received response from Gemini")

            output_file = os.path.join(output_folder, f"page_{page_num}_ocr.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            logger.info(f"Saved OCR results to {output_file}")

            is_toc = is_table_of_contents(extracted_text)
            logger.info(f"TOC detection result: {'YES' if is_toc else 'NO'}")

            if is_toc:
                logger.info(f"Found TOC on page {page_num}")
                file_name = pdfs_folder + "/input_page_" + str(page_num) + ".pdf"
                toc_page_list.append(file_name)
                toc_pages.append({"page_number": page_num, "text": extracted_text})

            if is_toc and not in_toc_section:
                in_toc_section = True
                toc_start_page = page_num
                logger.info(f"Table of contents detected starting at page {page_num}.")
            elif not is_toc and in_toc_section:
                logger.info(f"End of table of contents detected after page {page_num-1}.")
                toc_end_page = page_num - 1
                break

            logger.info(f"Waiting {DELAY_BETWEEN_REQUESTS} seconds before next request...")
            time.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")

    if not toc_pages:
        error_message = f"No table of contents was detected in the first {pages_to_check} pages of the document."
        logger.info(error_message)
    else:
        logger.info(f"TOC detection complete. Found {len(toc_pages)} TOC pages.")

    return toc_pages, toc_end_page, toc_page_list, error_message

def generate_json_toc(toc_list):
    """
    Generate a JSON representation of the table of contents from the OCR results.
    
    Args:
        toc_list (list): List of PDF file paths containing TOC pages.
        
    Returns:
        dict: JSON representation of the table of contents.
    """
    logger.info("Starting JSON TOC generation...")
    logger.info(f"Processing {len(toc_list)} TOC pages...")
    
    # Create a model instance
    model = setup_gemini(GEMINI_API_KEY)
    
    # Prepare the files for upload
    files = toc_list
    
    # Create the prompt
    prompt = """You are an OCR assistant specialized in extracting table of contents information from images or scanned PDF pages.

Your task is to:
1. PROCESS EVERY SINGLE PAGE provided without exception
2. Before starting, COUNT THE TOTAL NUMBER OF PAGES/IMAGES to ensure complete processing
3. IGNORE headings like "Table of Contents" or "Contents" - these are NOT actual entries
4. Extract EVERY entry visible across ALL pages - ensure complete extraction
5. Detect multi-level hierarchies with precision based PRIMARILY on indentation patterns
6. Pay SPECIAL ATTENTION to deeply indented entries that may be several levels deep
7. Identify chapter-specific authors when present
8. Accurately separate titles from page numbers even when they appear close together
9. Extract ONLY the page for each entry
10. For subsections without page numbers, inherit the page number from their parent section
11. Combine everything into a single, comprehensive JSON structure

CRITICAL INSTRUCTION FOR DEEP HIERARCHIES:
When you see entries that are significantly more indented than their parent, YOU MUST place them as children of the immediately preceding less-indented entry, no matter how large the indentation difference. DO NOT skip hierarchy levels even when indentation appears to "jump" several positions to the right.

HIERARCHY VERIFICATION PROCESS - CRITICAL:
* For EVERY entry, VERIFY its hierarchical relationship by checking:
  1. Its indentation level compared to surrounding entries
  2. Its logical connection to the preceding content
  3. The pattern of similar entries elsewhere in the document
* ALWAYS review the previous 3-5 entries when determining an entry's hierarchy level
* COMPARE each entry's indentation with all previous entries at various levels
* VALIDATE that entries with similar formatting and indentation are treated consistently
* If uncertain, TRACE upward through the hierarchy to verify correct parent-child relationships

DETAILED HIERARCHY DETECTION ALGORITHM - FOLLOW PRECISELY:
1. FIRST: Scan the entire document and identify ALL indentation positions
2. Sort these positions from least indented (left-most) to most indented (right-most)
3. For EACH entry, compare its indentation with preceding entries:
   * If an entry is more indented than the previous entry, it is a CHILD of that entry
   * If an entry is at the same indentation as a previous entry, they are SIBLINGS
   * If an entry is less indented than the previous entry, move up the hierarchy until finding an entry at the same indentation level - these are SIBLINGS
4. NEVER skip hierarchy levels - if an entry is significantly more indented, it is still a direct child of the preceding entry
5. For entries that appear to have multiple levels of indentation difference:
   * First identify which entry is its immediate parent (the closest less-indented entry above it)
   * Then nest it directly under that parent (even if this creates a large indentation jump)
   * NEVER create "empty" hierarchy levels just to maintain consistent indentation differences

SPECIAL INSTRUCTION FOR DEEPLY INDENTED SUBSECTIONS:
* Pay close attention to sections that may have significantly more indented subsections
* Even if subsections appear much more indented, they are DIRECT CHILDREN of their parent section
* The pattern might be: medium indentation → large indentation (skipping an "expected" indentation level)
* In these cases, DO NOT look for an "intermediate" indentation level - nest directly under the parent

HANDLING ENTRIES WITHOUT PAGE NUMBERS:
* If a subsection does not have its own page number, INHERIT the page number from its parent section
* NEVER leave page as null if a parent page number is available
* This inheritance applies to ALL levels of the hierarchy - deeply nested items without page numbers inherit from their closest ancestor with a page number

MULTI-PAGE DOCUMENT HANDLING:
1. Process EVERY page provided - do not stop after any particular page
2. Maintain hierarchy continuity across page boundaries
3. Apply the SAME indentation-to-level mapping across ALL pages
4. Before completing, verify that ALL pages have been fully processed
5. If the table of contents continues across multiple pages, maintain continuity of hierarchy
6. Check for any indication of continued content (e.g., "continued..." text)

RULES FOR SPECIAL CASES:
1. Sub-items following a main item are ALWAYS children of that item, NEVER siblings
2. Items with the same indentation are ALWAYS siblings at the same level
3. If an item appears under another item and is more indented, it is ALWAYS a child
4. Entries preceded by question numbers or similar prefixes are treated as distinct entries
5. Nested sub-points under such entries are children of those entries
6. Appendices and their content follow the same hierarchy rules as main chapters
7. Individual section titles under appendices are children of those appendices

CRITICAL EXTRACTION GUIDELINES:
1. Title and page number separation:
   * Carefully distinguish between title text and page numbers
   * Page numbers typically appear at the end of lines or entries
   * Page numbers are usually right-aligned or preceded by dots, dashes, or spaces
   * Even when page numbers appear very close to title text with no clear separator, identify them correctly
   * Page numbers should ONLY appear in the "page" field, never as part of the title

2. Chapter-specific author handling:
   * Identify when chapters have specific authors attributed to them
   * Distinguish chapter titles from author names
   * Author names typically follow chapter titles, often in different formatting
   * Author information may be prefixed with "by," "edited by," "contributed by," etc.
   * Place author information in a dedicated "author" field for each entry

3. APPENDIX AND SUPPLEMENTARY SECTION HANDLING:
   * Recognize that sections like "Appendix" are TOP-LEVEL entries (same level as chapters)
   * Treat content under appendices with the same hierarchy rules as main content
   * Pay special attention to hierarchy shifts after special sections like appendices, bibliographies, etc.

ANTI-HALLUCINATION REQUIREMENTS:
* Extract ONLY text that is clearly visible in the image(s)
* NEVER add entries that are not present in the provided image(s)
* If text is partially visible or unclear, indicate this with "[unclear]" rather than guessing
* Do not assume content beyond what is explicitly shown
* If a page number is not visible or provided, inherit from parent section
* Do not create hierarchical relationships that aren't visually apparent in the image(s)
* When in doubt about any element, omit it rather than risk including incorrect information

FORMAT RESPONSE AS VALID JSON:
{
  "tableOfContents": [
    {
      "number": "PREFIX",  // Chapter/section number/prefix if present
      "title": "TITLE",    // The actual title text (WITHOUT page numbers)
      "author": "AUTHOR",  // Author name if specifically attributed to this entry (null if none)
      "page": X,      // Starting page number (inherit from parent if not visible)
      "items": [
        // Child entries with identical structure
        // Can be nested to ANY depth required by the document
        // NEVER break hierarchical chains
      ]
    }
  ]
}

MANDATORY VERIFICATION STEPS - COMPLETE ALL THESE BEFORE FINALIZING:
1. Deep Indentation Check: Verify deeply indented items are properly nested under their parent sections
2. Complete Page Processing: Verify ALL pages have been processed
3. Indentation-Based Hierarchy: Confirm entries are nested based on their relative indentation
4. Parent-Child Relationships: Confirm more indented entries are children of their nearest less-indented parent
5. Sibling Relationships: Verify entries with identical indentation are siblings
6. Sub-Section Verification: Double-check all sub-sections are properly nested under their correct parent sections
7. Multi-Level Nesting: Verify that multi-level indentation is properly represented in the nesting structure
8. Page Inheritance: Confirm subsections without page numbers inherit page numbers from their parent sections
9. Header Exclusion: Verify "Table of Contents" or "Contents" headings are NOT included as entries
10. Title-Page Number Separation: Confirm title text is completely separate from page numbers

FINAL OUTPUT REQUIREMENTS:
- Process EVERY SINGLE PAGE provided - count them at the start
- Extract EVERY entry visible across ALL pages
- Preserve the EXACT hierarchy based on visual indentation patterns
- Correctly nest deeply indented entries under their appropriate parent
- Include ONLY the page (no endPage or pageCount)
- Inherit page numbers for subsections without their own page numbers
- Preserve exact title text (removing only page numbers and separators)
- NEVER include page numbers as part of the title text
- Maintain ALL hierarchical relationships exactly as they appear in the document
- Support UNLIMITED nesting depth - preserve ALL levels of hierarchy
- Return ONLY the valid JSON structure without any explanations"""

    logger.info("Loading PDF files...")
    # Load the PDF files
    contents = []
    for i, file_path in enumerate(files, 1):
        logger.info(f"Loading PDF file {i}/{len(files)}: {os.path.basename(file_path)}")
        with open(file_path, "rb") as f:
            contents.append({"mime_type": "application/pdf", "data": f.read()})
    
    logger.info("Setting up Gemini generation config...")
    # Set generation config
    generation_config = GenerationConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
    )
    
    logger.info("Sending request to Gemini for JSON generation...")
    # Generate content
    response = model.generate_content(
        contents=[prompt] + contents,
        generation_config=generation_config,
        stream=True
    )
    
    logger.info("Receiving response from Gemini...")
    json_string = ""
    for chunk in response:
        json_string += chunk.text
    
    logger.info("Response received from Gemini")

    # Remove Markdown code block markers if present
    if json_string.startswith("```"):
        logger.info("Cleaning JSON response...")
        # Find the end of the first line and skip it
        first_line_end = json_string.find("\n")
        if first_line_end != -1:
            # Find the closing code block
            closing_markers = json_string.rfind("```")
            if closing_markers > first_line_end:
                # Extract just the JSON content without the markers
                json_string = json_string[first_line_end+1:closing_markers].strip()
            else:
                # Only remove opening markers if no closing found
                json_string = json_string[first_line_end+1:].strip()
    
    logger.info("Parsing JSON response...")
    # Now parse the cleaned JSON
    try:
        toc_json = json.loads(json_string)
        logger.info("JSON parsing complete!")
        return toc_json
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        # Return a minimal valid structure
        return {"tableOfContents": []}

def process_images_start_page(folder_path, output_folder, model, toc_end_page, toc_json, manual_offset=None, search_limit=15):
    """
    Finds the PDF page number where the printed page "1" appears, using OCR and Gemini's YES/NO response.
    Args:
        folder_path (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where OCR output is saved.
        model: The Gemini model.
        toc_end_page (int): The PDF page number where the table of contents ends.
        toc_json (dict): The TOC JSON, containing titles and printed page numbers.
        manual_offset (int, optional): A manual offset value provided by the user. If provided, the function skips the OCR search.
        search_limit (int): The maximum number of pages to search for page "1" after the TOC.

    Returns:
        int: The calculated offset between printed page numbers and PDF page numbers, or None if not found.
    """
    logger.info("Starting process_images_start_page function")
    if manual_offset is not None:
        logger.info(f"Manual offset provided: {manual_offset}")
        return manual_offset

    if toc_end_page is not None:
        start_page = toc_end_page + 1
        logger.info(f"Starting search from TOC end page + 1: {start_page}")
    else:
        start_page = 1
        logger.info("No TOC end page. Starting search from page 1")

    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    #  Count total number of image files with supported formats
    total_images = len([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and
           Path(f).suffix.lower() in supported_formats
    ])

    if total_images == 0:
        logging.error(f"No supported image files found in {folder_path}")
        return 0  # Default to zero offset if no images

    logging.info(f"Found {total_images} images to process")

    # Check if tableOfContents exists and is not empty
    if "tableOfContents" not in toc_json or not toc_json["tableOfContents"]:
        logging.warning("tableOfContents is empty or not found in toc_json.")
        return 0  # Default to zero offset

    first_item = toc_json["tableOfContents"][0]
    title = first_item.get("title", "").lower()
    printed_page = str(first_item.get("page", "1"))  # Default to "1" if not found

    logger.info(f"Searching for title: '{title}' and page: '{printed_page}'")

    for page_num in range(start_page, min(start_page + search_limit, total_images + 1)):
        logger.info(f"Processing PDF page: {page_num}")
        image_file = f"page_{page_num}.png"
        image_path = os.path.join(folder_path, image_file)

        if not os.path.exists(image_path):
            found = False
            for ext in supported_formats:
                alt_file = f"page_{page_num}{ext}"
                alt_path = os.path.join(folder_path, alt_file)
                if os.path.exists(alt_path):
                    image_file = alt_file
                    image_path = alt_path
                    found = True
                    break

            if not found:
                logging.warning(f"Could not find image file for page {page_num}, skipping")
                continue

        try:
            img = Image.open(image_path)

            # Modified Prompt
            prompt_template = f"""TASK: Page 1 and Title Confirmation
SYSTEM OBJECTIVE: Determine if the specified title "{title}" and page number "{printed_page}" are present on the given image.

INPUT: A single PDF page image.

OUTPUT FORMAT:
- FIRST LINE MUST BE ONLY "YES" or "NO" (capitalized).
- Following lines should contain your reasoning.

DETECTION CRITERIA:
- Explicitly confirm the presence of both the title "{title}" AND the page number "{printed_page}" on the page.

REASONING PROCESS:
1. Extract all text from the provided image.
2. Check for the presence of "{title}" in the extracted text (case-insensitive).
3. Check for the presence of "{printed_page}" in the extracted text, considering potential variations (e.g., surrounded by whitespace).
4. If both conditions are met, respond with "YES" on the first line, followed by your reasoning.
5. If either condition is not met, respond with "NO" on the first line, followed by your reasoning.

IMPORTANT: Provide a concise explanation of your reasoning in the following lines, specifying whether each element (title and page number) was found.
"""
            response = model.generate_content([prompt_template, img])
            extracted_text = response.text
            first_line = extracted_text.strip().split('\n')[0].strip().upper()

            logger.info(f"Gemini's response: {first_line}")

            if first_line == "YES":
                logging.info(f"Gemini confirmed title '{title}' and page '{printed_page}' on PDF page {page_num}")
                offset = page_num - int(printed_page)
                logger.info(f"Offset found: {offset}")
                return offset
            else:
                logging.info(f"Gemini did not find title and page on PDF page {page_num}")

        except Exception as e:
            logging.error(f"Error processing page {page_num}: {str(e)}")

    logging.warning("Title and Printed page 1 not found within search limit.")
    # Return default offset if search fails
    return 0

def update_toc_json_with_page_numbers(toc_json, folder_path, offset):
    """
    Updates the table of contents JSON with startPage, endPage, and totalPages values
    using a for loop and creating a new list.

    Args:
        toc_json (dict): The table of contents JSON object.
        folder_path (str): Path to the folder containing the images (for total_images).
        offset (int): The offset between printed page numbers and PDF page numbers.

    Returns:
        dict: The updated table of contents JSON object.
    """
    logger.info(f"Updating TOC JSON with offset: {offset}")
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']  # Define supported formats

    # Count total number of image files with supported formats
    total_images = len([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and
           Path(f).suffix.lower() in supported_formats
    ])

    if total_images == 0:
        logging.error(f"No supported image files found in {folder_path}")
        return toc_json  # Return original if no images found

    logging.info(f"Found {total_images} images to process")

    new_list = []
    toc_items = toc_json.get("tableOfContents", [])  # Safely get the list

    num_items = len(toc_items) #get the number of toc items
    logger.info(f"Processing {num_items} TOC items")

    for each_index in range(num_items):
        new_item = toc_items[each_index].copy() #copy the original item
        # Calculate startPage
        new_item['startPage'] = new_item.get("page", 1) + offset - 1
        # Keep the 'page' field for reference
        
        # Calculate endPage:
        if each_index < num_items - 1:  # Not the last item
            next_item = toc_items[each_index + 1]
            new_item['endPage'] = next_item.get("page", 1) + offset - 2 # the item before end page -1
        else:  # Last item
            new_item['endPage'] = total_images - 1 #zero indexed

        new_item['totalPages'] = new_item['endPage'] - new_item['startPage'] + 1 #calculate total

        # Process nested items recursively if they exist
        if 'items' in new_item and new_item['items']:
            # Create a temporary structure for recursive processing
            temp_json = {"tableOfContents": new_item['items']}
            updated_temp = update_toc_json_with_page_numbers(temp_json, folder_path, offset)
            new_item['items'] = updated_temp.get("tableOfContents", [])

        new_list.append(new_item) #append item to new array

    toc_json['tableOfContents'] = new_list #assign array
    return toc_json

def cleanup_temp_files(folder_path):
    """
    Clean up temporary files created during processing and remove the folder itself.
    
    Args:
        folder_path (str): Path to the folder containing temporary files.
    """
    logger.info(f"Cleaning up temporary files in {folder_path}")
    try:
        if os.path.exists(folder_path):
            # Use shutil.rmtree to remove the directory and all its contents in one go
            import shutil
            shutil.rmtree(folder_path)
            logger.info(f"Successfully removed directory and all contents: {folder_path}")
        else:
            logger.warning(f"Cleanup folder {folder_path} does not exist")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")

def save_processed_files(filename, pdf_path, json_data):
    """
    Create a unique folder for this run and save both the PDF and JSON data
    
    Args:
        filename (str): Original filename
        pdf_path (str): Path to the PDF file
        json_data (dict): JSON data to save
        
    Returns:
        tuple: (folder_path, pdf_path, json_path)
    """
    # Create a unique folder name using timestamp and filename
    base_name = os.path.splitext(filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder_name = f"{base_name}_{timestamp}"
    
    # Create the folder path
    folder_path = os.path.join(app.config['AI_PDF_DATA'], unique_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Save the PDF with its original filename
    target_pdf_path = os.path.join(folder_path, filename)
    json_path = os.path.join(folder_path, "toc.json")
    
    try:
        # Copy the PDF
        import shutil
        shutil.copy2(pdf_path, target_pdf_path)
        logger.info(f"Saved PDF to {target_pdf_path}")
        
        # Save the JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved JSON data to {json_path}")
        
        return folder_path, target_pdf_path, json_path
    except Exception as e:
        logger.error(f"Error saving processed files: {str(e)}")
        return None, None, None

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Split-PDF')
def split_pdf_local():
    return render_template('split_pdf_local.html')

@app.route('/Split-PDF/output/<filename>')
def split_pdf_local_output(filename):
    """Direct rendering of the output page without file upload"""
    return render_template('pdf_local_split_output.html', filename=filename)

@app.route('/get_pdf_data/<filename>')
def get_pdf_data(filename):
    """Provide PDF data for client-side processing"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        try:
            # Return the file with appropriate content type for direct access
            return send_file(filepath, 
                           mimetype='application/pdf', 
                           as_attachment=False,
                           download_name=filename)
        except Exception as e:
            logger.error(f"Error serving PDF file: {str(e)}")
            return "Error serving file", 500
    return "File not found", 404

@app.route('/upload-local', methods=['POST'])
def upload_local():
    """
    Handle local PDF splitting which is completely client-side.
    No files are saved to the server - all processing happens in the browser.
    This just validates the file and returns the template.
    """
    # We don't need to save the file - the client will handle it completely via IndexedDB
    if 'pdf_file' not in request.files:
        flash('No file part')
        return redirect(url_for('split_pdf_local'))
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('split_pdf_local'))
    
    if file and file.filename.endswith('.pdf'):
        # IMPORTANT: We only pass the filename to the template without saving anything on the server
        # The actual PDF data is handled entirely in the browser using JavaScript and IndexedDB
        return render_template('pdf_local_split_output.html', 
                              filename=file.filename)
    else:
        flash('Invalid file type. Please upload a PDF file.')
        return redirect(url_for('split_pdf_local'))

@app.route('/AI-PDF-Splitter')
def split_pdf_ai():
    return render_template('split_pdf_ai.html')

@app.route('/get_original_pdf/<filename>')
def get_original_pdf(filename):
    """
    Find and serve a PDF file that was processed with AI.
    First checks in all subdirectories of AI-PDF-Data for the original filename 
    in a folder that starts with the base name of the requested file.
    Falls back to direct files in AI-PDF-Data or uploads directory.
    """
    base_name = os.path.splitext(filename)[0]
    
    # First check all subdirectories in AI-PDF-Data for matching folders
    if os.path.exists(app.config['AI_PDF_DATA']):
        for folder_name in os.listdir(app.config['AI_PDF_DATA']):
            folder_path = os.path.join(app.config['AI_PDF_DATA'], folder_name)
            
            # If it's a directory and starts with the base filename
            if os.path.isdir(folder_path) and folder_name.startswith(base_name + "_"):
                # Check for the original filename
                pdf_path = os.path.join(folder_path, filename)
                if os.path.exists(pdf_path):
                    logger.info(f"Found PDF in folder structure: {pdf_path}")
                    return send_file(pdf_path, mimetype='application/pdf')
                
                # For backward compatibility, also check for "original.pdf"
                pdf_path = os.path.join(folder_path, "original.pdf")
                if os.path.exists(pdf_path):
                    logger.info(f"Found PDF (as original.pdf) in folder structure: {pdf_path}")
                    return send_file(pdf_path, mimetype='application/pdf')
    
    # If not found in folder structure, try direct file (backward compatibility)
    filepath = os.path.join(app.config['AI_PDF_DATA'], filename)
    if os.path.exists(filepath):
        logger.info(f"Found PDF directly in AI-PDF-Data: {filepath}")
        return send_file(filepath, mimetype='application/pdf')
    
    # Finally check the uploads directory
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        logger.info(f"Found PDF in uploads directory: {filepath}")
        return send_file(filepath, mimetype='application/pdf')
    
    logger.warning(f"PDF file not found: {filename}")
    return "File not found", 404

@app.route('/upload', methods=['POST'])
def upload_post():
    # Log that we've started processing an upload
    logger.info("Upload request received")
    
    # Debug information about the request
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request content type: {request.content_type}")
    logger.debug(f"Request form keys: {list(request.form.keys())}")
    logger.debug(f"Request files keys: {list(request.files.keys())}")
    logger.debug(f"Request headers: {dict(request.headers)}")
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set. Please set the environment variable.")
        flash('GEMINI_API_KEY not set. Please set the environment variable.')
        return redirect(url_for('split_pdf_ai'))
    
    # Check if the post request has the file part
    if 'pdf_file' not in request.files:
        logger.error("No 'pdf_file' in request.files. Available keys: " + str(list(request.files.keys())))
        flash('No file part found in the upload. Please try again.')
        return redirect(url_for('split_pdf_ai'))
    
    file = request.files['pdf_file']
    logger.info(f"File in request: name='{file.filename}', content_type={file.content_type}")
    
    if file.filename == '':
        logger.error("No filename in uploaded file")
        flash('No selected file')
        return redirect(url_for('split_pdf_ai'))
    
    if not file.filename.endswith('.pdf'):
        logger.error(f"File {file.filename} is not a PDF")
        flash('Invalid file type. Please upload a PDF file.')
        return redirect(url_for('split_pdf_ai'))
    
    # Make sure upload directory exists again (just in case)
    try:
        # Check permissions first
        upload_dir_status = check_directory_permissions(app.config['UPLOAD_FOLDER'])
        if upload_dir_status["error"]:
            raise PermissionError(f"Upload directory issue: {upload_dir_status['error']}")
            
        # Create directory if needed
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True, mode=0o755)
        
        # Save uploaded file to temporary location with safe filename
        from werkzeug.utils import secure_filename
        safe_filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        logger.info(f"Attempting to save file to: {filepath}")
        
        try:
            # Read and save the file content in one operation
            file.save(filepath)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not saved at {filepath}")
            
            file_size = os.path.getsize(filepath)
            logger.info(f"File saved successfully at {filepath}. Size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("File was saved but is empty (0 bytes). Check upload process.")

        except IOError as e:
            logger.exception(f"I/O error saving file: {str(e)}")
            flash(f"Unable to save file: {str(e)}. Check server permissions.")
            return redirect(url_for('split_pdf_ai'))
    except Exception as e:
        logger.exception(f"Unexpected error saving file: {str(e)}")
        flash(f"Error saving file: {str(e)}")
        return redirect(url_for('split_pdf_ai'))
    
    try:
        # Create unique folders for processing
        folders = create_unique_folders(filepath, app.config['UPLOAD_FOLDER'])
        
        # Split PDF into individual pages
        split_pdf(filepath, folders["split_pdf"], individual_pages=True)
        
        # Convert PDF to images
        convert_pdf_to_images(filepath, output_dir=folders["images"])
        
        # Initialize Gemini model
        model = setup_gemini(GEMINI_API_KEY)
        
        # Process images to detect table of contents
        toc_pages, toc_end_page, toc_page_list, error_message = process_images_toc(
            folders["images"], 
            folders["ocr_output"], 
            folders["split_pdf"], 
            model
        )
        
        if error_message:
            # Clean up temporary folders since we're returning early
            cleanup_temp_files(folders["root"])
            flash(error_message)
            return redirect(url_for('split_pdf_ai'))
            
        if not toc_page_list:
            # Clean up temporary folders since we're returning early
            cleanup_temp_files(folders["root"])
            logger.warning("No table of contents found in document.")
            flash('No table of contents detected in the uploaded PDF. Please ensure the document contains a table of contents.')
            return redirect(url_for('split_pdf_ai'))
        
        # Generate TOC JSON
        toc_json = generate_json_toc(toc_page_list)
        
        # Find offset between printed page numbers and PDF page numbers
        offset = process_images_start_page(
            folders["images"], 
            folders["ocr_output"], 
            model, 
            toc_end_page, 
            toc_json
        )
        
        # Update JSON with calculated page numbers
        updated_toc_json = update_toc_json_with_page_numbers(toc_json, folders["images"], offset)
        
        # Save the processed files
        folder_path, target_pdf_path, json_path = save_processed_files(safe_filename, filepath, updated_toc_json)
        
        # Now that we have the final data, clean up the temporary processing folders and original upload
        cleanup_temp_files(folders["root"])
        
        # Remove the original temporary file after it's been copied to the AI-PDF-Data directory
        if target_pdf_path and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Removed temporary upload: {filepath}")
            except Exception as e:
                logger.warning(f"Could not remove temporary upload file: {str(e)}")
        
        # Get just the folder name for display/reference
        folder_name = os.path.basename(folder_path) if folder_path else None
        
        return render_template('pdf_ai_output.html', 
                              json_data=updated_toc_json, 
                              filename=safe_filename,
                              folder_name=folder_name)
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Try to clean up any created folders in case of exception
        if 'folders' in locals() and 'root' in folders:
            cleanup_temp_files(folders["root"])
        # Try to remove the uploaded file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        flash(f'Error processing PDF: {str(e)}')
        return redirect(url_for('split_pdf_ai'))

@app.route('/get_toc_json/<filename>')
def get_toc_json(filename):
    """
    Find and serve the TOC JSON data for a processed PDF file.
    Follows the same folder structure as get_original_pdf.
    """
    base_name = os.path.splitext(filename)[0]
    
    # First check all subdirectories in AI-PDF-Data for matching folders
    if os.path.exists(app.config['AI_PDF_DATA']):
        for folder_name in os.listdir(app.config['AI_PDF_DATA']):
            folder_path = os.path.join(app.config['AI_PDF_DATA'], folder_name)
            
            # If it's a directory and starts with the base filename
            if os.path.isdir(folder_path) and folder_name.startswith(base_name + "_"):
                # Check for toc.json
                json_path = os.path.join(folder_path, "toc.json")
                if os.path.exists(json_path):
                    logger.info(f"Found TOC JSON in folder structure: {json_path}")
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    return jsonify(json_data)
    
    # If not found in folder structure, try direct file (backward compatibility)
    json_path = os.path.join(app.config['AI_PDF_DATA'], f"{base_name}_toc.json")
    if os.path.exists(json_path):
        logger.info(f"Found TOC JSON directly in AI-PDF-Data: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return jsonify(json_data)
    
    logger.warning(f"TOC JSON file not found for: {filename}")
    return jsonify({"error": "TOC data not found"}), 404

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Flask PDF Table of Contents Generator')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the application on')
    args = parser.parse_args()
    
    app.run(debug=True, port=args.port) 
