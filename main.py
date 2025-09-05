import os
import httpx
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import asyncio
import time
import logging
from playwright.async_api import async_playwright
import re
from dotenv import load_dotenv
load_dotenv()
from typing import Generator, List, Optional
from datetime import datetime

# Database (SQLAlchemy)
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY_BASE = 1.0  # Base delay between retries (doubles each time)

# Rate limiting
last_api_call = 0
min_interval = 1.0  # Minimum 1 second between API calls

# Validate API key is present
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable is not set!")
    print("Please set it with: export GEMINI_API_KEY='your_api_key_here'")
    print("Or create a .env file with: GEMINI_API_KEY=your_api_key_here")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def clean_html_output(html_content):
    """Clean HTML output from AI model to remove markdown formatting and ensure clean HTML"""
    print(html_content)
    if not html_content:
        return html_content
    
    # Remove markdown code blocks
    import re
    
    # Remove ```html and ``` blocks
    html_content = re.sub(r'```html\s*', '', html_content)
    html_content = re.sub(r'```\s*$', '', html_content)
    
    # Remove any leading/trailing whitespace
    html_content = html_content.strip()
    
    # Ensure the content starts with < and ends with >
    if not html_content.startswith('<'):
        # Find the first < character
        start_idx = html_content.find('<')
        if start_idx != -1:
            html_content = html_content[start_idx:]
    
    if not html_content.endswith('>'):
        # Find the last > character
        end_idx = html_content.rfind('>')
        if end_idx != -1:
            html_content = html_content[:end_idx + 1]
    
    # Remove any remaining markdown artifacts
    html_content = re.sub(r'^```.*?\n', '', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'\n```$', '', html_content, flags=re.MULTILINE)
    
    # Remove any explanation text that might be before or after HTML
    lines = html_content.split('\n')
    html_lines = []
    in_html = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('<'):
            in_html = True
        if in_html:
            html_lines.append(line)
        if line.endswith('>') and in_html:
            # Check if this might be the end of HTML content
            if any(tag in line for tag in ['</html>', '</body>', '</div>']):
                in_html = False
    
    cleaned_html = '\n'.join(html_lines)
    
    # Final cleanup - ensure we have valid HTML
    if not cleaned_html.startswith('<'):
        return html_content  # Return original if cleaning failed
    
    return cleaned_html

def extract_dimensions_from_template(template):
    """Extract width and height from HTML template size comments"""
    
    # Look for size comments like <!--SIZE: 970x250-->
    size_pattern = r'<!--SIZE:\s*(\d+)x(\d+)-->'
    match = re.search(size_pattern, template)
    
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    
    # Fallback: Look for size classes like size-300x250, size-728x90, etc.
    size_pattern_fallback = r'size-(\d+)x(\d+)'
    match_fallback = re.search(size_pattern_fallback, template)
    
    if match_fallback:
        width = int(match_fallback.group(1))
        height = int(match_fallback.group(2))
        return width, height
    
    # Default dimensions if no size information found
    return 1200, 628


############################################
# Database setup (PostgreSQL via SQLAlchemy)
############################################

# DATABASE_URL should be provided like:
# postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME
# Fallback to local sqlite for dev if not provided
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


class Template(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    platform = Column(String(50), nullable=True, index=True)
    size_name = Column(String(50), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    html = Column(Text, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    template_id = Column(Integer, nullable=True, index=True)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=True)
    template_html = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def serialize_template(t: Template) -> dict:
    return {
        "id": t.id,
        "name": t.name,
        "platform": t.platform,
        "size_name": t.size_name,
        "width": t.width,
        "height": t.height,
        "is_default": t.is_default,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "html": t.html,
    }


def serialize_chat_history(ch: ChatHistory) -> dict:
    return {
        "id": ch.id,
        "session_id": ch.session_id,
        "template_id": ch.template_id,
        "user_message": ch.user_message,
        "ai_response": ch.ai_response,
        "template_html": ch.template_html,
        "created_at": ch.created_at.isoformat() if ch.created_at else None,
    }


@app.on_event("startup")
async def on_startup():
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Seed default template if table is empty or no default exists
    db = SessionLocal()
    try:
        has_any = db.query(Template).first()
        if not has_any:
            # Create a default template using the base template generator
            default_template = generate_base_template(1200, 628)
            width, height = extract_dimensions_from_template(default_template)
            default_row = Template(
                name=f"Default {width}x{height}",
                platform="Generic",
                size_name=f"{width}x{height}",
                width=width,
                height=height,
                html=default_template,
                is_default=True,
            )
            db.add(default_row)
            db.commit()
        else:
            default_exists = db.query(Template).filter(Template.is_default == True).first()
            if not default_exists:
                # Create a default template using the base template generator
                default_template = generate_base_template(1200, 628)
                width, height = extract_dimensions_from_template(default_template)
                default_row = Template(
                    name=f"Default {width}x{height}",
                    platform="Generic",
                    size_name=f"{width}x{height}",
                    width=width,
                    height=height,
                    html=default_template,
                    is_default=True,
                )
                db.add(default_row)
                db.commit()
    finally:
        db.close()


async def call_gemini(template, prompt):
    # Check if API key is available
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
        )
    
    # Rate limiting
    global last_api_call
    current_time = time.time()
    if current_time - last_api_call < min_interval:
        sleep_time = min_interval - (current_time - last_api_call)
        logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds before API call")
        await asyncio.sleep(sleep_time)
    
    api_prompt = (
         "You are a professional HTML email/banner creative generator. "
    "Follow these rules strictly: "
    "- DO NOT remove any comment or size comment. "
    "- no animation no effects no hover effects no transitions"
    "- ONLY do exactly what the user instructs. "
    "- If the user asks for resizing or repositioning, adjust elements only according to the specified template size. "
    "- If there is unavoidable empty space, fill it with minimal appropriate design elements (background color, pattern, or spacing). "
    "- Do not add or remove tags unless the user explicitly tells you. "
    "- Use the exact image URLs provided, without validation. "
    "- If no image is provided browse public urls for that image"
    "- All images must use object-fit: contain. "
    "- If the user asks for a new design or redo, completely change the layout with a modern UI, "
    "but keep the same size and content. Use placeholder images for redesigns. "
    "- Always make elements fit the given size proportionally. "
    "- sometimes user will give wage instruction, you need to think with UI/UX prespective and can change the design. for example, if user says to add a logo, add it. or user says add text under the headline, then check what size of the text will be fit in the space and add it. Think for all the asp"
    "- Return ONLY the final HTML code. Do not add explanations, notes, or extra text.\n\n"
    f"Template:\n{template}\n"
    f"Creative Brief:\n{prompt}\n"
    )

    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": api_prompt}
                ]
            }
        ]
    }

    params = {"key": GEMINI_API_KEY}

    max_retries = MAX_RETRIES
    retry_delay = RETRY_DELAY_BASE

    for i in range(max_retries):
        try:
            # Update last API call time
            last_api_call = time.time()
            logger.info(f"Attempting Gemini API call (attempt {i+1}/{max_retries})")
            
            # Use a longer timeout for the Gemini API
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers, params=params)
                resp.raise_for_status()
                gemini_output = resp.json()
                logger.info("Gemini API call successful")
                # Parse model response (if needed, adjust this depending on Gemini output)
                for candidate in gemini_output.get("candidates", []):
                    html = candidate["content"]["parts"][0]["text"]
                    if "<" in html and ">" in html:
                        # Clean the HTML output to remove markdown formatting
                        cleaned_html = clean_html_output(html)
                        logger.info("HTML cleaned successfully")
                        return cleaned_html
                raise ValueError("No valid HTML found in Gemini output.")
        except httpx.TimeoutException:
            if i < max_retries - 1:
                logger.warning(f"Gemini API request timed out. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("All retries exhausted for timeout.")
                raise HTTPException(status_code=408, detail="Request timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.error(f"Gemini API authentication failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API authentication failed. Please check your API key. Error: {e}"
                )
            elif e.response.status_code == 404:
                logger.error(f"Gemini API endpoint not found: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API endpoint not found. Please check the model name. Error: {e}"
                )
            elif e.response.status_code == 503:
                if i < max_retries - 1:
                    logger.warning(f"Gemini API returned 503 (Service Unavailable). Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("All retries exhausted for 503 error.")
                    raise HTTPException(status_code=503, detail="The AI service is temporarily unavailable. Please try again in a few minutes.")
            else:
                logger.error(f"Gemini API HTTP error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error: {e}"
                )
        except Exception as e:
            if i < max_retries - 1:
                logger.warning(f"Error calling Gemini API. Retrying in {retry_delay} seconds... Error: {str(e)}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"All retries exhausted. Last error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate content: {str(e)}")

async def html_to_image(html_content, width=None, height=None, scale=2):
    """Convert HTML to image with specified dimensions"""
    # Clean and validate HTML content
    if not html_content or not isinstance(html_content, str):
        raise ValueError("Invalid HTML content provided")
    
    # Ensure HTML content is clean (no markdown formatting)
    cleaned_html = clean_html_output(html_content)
    
    # Validate that we have valid HTML
    if not cleaned_html.startswith('<') or not cleaned_html.endswith('>'):
        raise ValueError("Invalid HTML format - content must be valid HTML")
    
    # If dimensions not provided, extract from HTML template
    if width is None or height is None:
        width, height = extract_dimensions_from_template(cleaned_html)
    
    logger.info(f"Generating image with dimensions: {width}x{height}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set viewport to match template dimensions
        await page.set_viewport_size({'width': width, 'height': height})
        await page.set_content(cleaned_html)
        
        # Wait for any dynamic content to load
        await page.wait_for_load_state('networkidle')
        
        tempf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        await page.screenshot(path=tempf.name, full_page=False)  # Use viewport size instead of full page
        await browser.close()
        return tempf.name

# Available creative sizes
AVAILABLE_SIZES = {
    "Facebook": [
        {"name": "Square", "width": 1080, "height": 1080},
        {"name": "Story", "width": 1080, "height": 1920},
        {"name": "Feed", "width": 1200, "height": 628}
    ],
    "Instagram": [
        {"name": "Square", "width": 1080, "height": 1080},
        {"name": "Story", "width": 1080, "height": 1920}
    ],
    "Google": [
        {"name": "Ad", "width": 125, "height": 125}
    ]
}

def generate_base_template(width, height):
    """Generate a base HTML template with specified dimensions"""
    return f"""<!doctype html>
<!--SIZE: {width}x{height}-->
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Generated Creative</title>
  <style>
    :root {{
      --w: {width}px; 
      --h: {height}px;
      --brand: #4f46e5;
      --accent: #06b6d4;
      --ink: #0b1021;
      --muted: #475569;
      --bg: #f8fafc;
      --cta-color: #ffffff;
      --radius: 14px;
      --shadow: 0 8px 24px rgba(2,6,23,0.18);
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }}

    body {{
      margin: 0;
      padding: 0;
      background: transparent;
    }}

    .creative {{
      width: var(--w);
      height: var(--h);
      background: var(--bg);
      position: relative;
      overflow: hidden;
      box-sizing: border-box;
    }}

    .content {{
      position: absolute;
      inset: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
    }}

    .headline {{
      font-size: {min(width // 15, 48)}px;
      font-weight: 800;
      color: var(--ink);
      margin: 0 0 16px 0;
      line-height: 1.2;
    }}

    .subtitle {{
      font-size: {min(width // 25, 24)}px;
      color: var(--muted);
      margin: 0 0 24px 0;
      line-height: 1.4;
    }}

    .cta {{
      background: linear-gradient(135deg, var(--brand), var(--accent));
      color: var(--cta-color);
      padding: 12px 24px;
      border-radius: var(--radius);
      text-decoration: none;
      font-weight: 600;
      font-size: {min(width // 30, 18)}px;
      border: none;
      cursor: pointer;
      box-shadow: var(--shadow);
    }}

    .logo {{
      position: absolute;
      top: 20px;
      left: 20px;
      width: {min(width // 10, 60)}px;
      height: {min(width // 10, 60)}px;
      background: linear-gradient(135deg, var(--brand), var(--accent));
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .logo svg {{
      width: 60%;
      height: 60%;
      fill: white;
    }}
  </style>
</head>
<body>
  <div class="creative">
    <div class="logo">
      <svg viewBox="0 0 24 24">
        <path d="M12 2v6M12 16v6M2 12h6M16 12h6M5 5l4 4M15 15l4 4M5 19l4-4M15 9l4-4"/>
      </svg>
    </div>
    <div class="content">
      <h1 class="headline">Your Headline Here</h1>
      <p class="subtitle">Your subtitle or description goes here</p>
      <button class="cta">Call to Action</button>
    </div>
  </div>
</body>
</html>"""

async def generate_template_from_scratch(prompt, size_name, platform):
    """Generate a complete template from scratch using AI"""
    
    # Find the size dimensions
    size_config = None
    for size in AVAILABLE_SIZES.get(platform, []):
        if size["name"] == size_name:
            size_config = size
            break
    
    if not size_config:
        raise HTTPException(status_code=400, detail=f"Invalid size '{size_name}' for platform '{platform}'")
    
    width, height = size_config["width"], size_config["height"]
    
    # Generate base template
    base_template = generate_base_template(width, height)
    
    # Create AI prompt for template generation
    ai_prompt = f"""You are a professional creative designer. Create a complete HTML banner ad with the following requirements:

Platform: {platform}
Size: {width}x{height} pixels
Creative Brief: {prompt}

Requirements:
- Create a modern, visually appealing design
- Use the exact dimensions {width}x{height}
- Include appropriate typography that scales with the size
- Add visual elements like backgrounds, shapes, or patterns
- Include a compelling headline and call-to-action
- Use CSS for styling (no external dependencies)
- Ensure the design is responsive within the given dimensions
- Make it look professional and engaging

Return ONLY the complete HTML code with embedded CSS. Do not include any explanations or markdown formatting."""

    try:
        # Call Gemini API to generate the template
        html = await call_gemini(base_template, ai_prompt)
        return html
    except Exception as e:
        logger.error(f"Error generating template from scratch: {str(e)}")
        # Return base template if AI generation fails
        return base_template


@app.post("/generate-image/")
async def generate_image(
    template: str = Form(...),
    prompt: str = Form(...),
):
    try:
        # Extract dimensions from the template
        width, height = extract_dimensions_from_template(template)
        logger.info(f"Extracted dimensions from template: {width}x{height}")
        
        # Pass the prompt directly to call_gemini
        html = await call_gemini(template, prompt)
        img_path = await html_to_image(html, width, height)
        return FileResponse(img_path, media_type='image/png')
    except HTTPException as e:
        # Re-raise HTTP exceptions as-is
        raise e
    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Provide more specific error messages based on the error type
        if "503" in str(e) or "Service Unavailable" in str(e):
            raise HTTPException(
                status_code=503, 
                detail="The AI service is temporarily unavailable. Please try again in a few minutes."
            )
        elif "timeout" in str(e).lower():
            raise HTTPException(
                status_code=408,
                detail="Request timed out. Please try again."
            )
        elif "authentication" in str(e).lower() or "403" in str(e):
            raise HTTPException(
                status_code=500,
                detail="Authentication error with AI service. Please contact support."
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"An unexpected error occurred while generating the image. Please try again."
            )


@app.get("/template/dimensions")
async def get_template_dimensions(template: str = None):
    """Get the dimensions of a template"""
    if template is None:
        # Get default template from database
        db = SessionLocal()
        try:
            default_template = db.query(Template).filter(Template.is_default == True).first()
            if not default_template:
                default_template = db.query(Template).first()
            if not default_template:
                raise HTTPException(status_code=404, detail="No templates found in database")
            template = default_template.html
        finally:
            db.close()
    
    width, height = extract_dimensions_from_template(template)
    return {
        "width": width,
        "height": height,
        "size_class": f"size-{width}x{height}"
    }

@app.get("/template/default")
async def get_default_template():
    """Get the default template from DB as JSON"""
    db = SessionLocal()
    try:
        row = db.query(Template).filter(Template.is_default == True).first()
        if not row:
            row = db.query(Template).first()
        if not row:
            raise HTTPException(status_code=404, detail="No templates found")
        return {
            "template": row.html,
            "meta": {
                "id": row.id,
                "name": row.name,
                "platform": row.platform,
                "size_name": row.size_name,
                "width": row.width,
                "height": row.height,
                "is_default": row.is_default,
            },
        }
    finally:
        db.close()


@app.get("/templates")
async def list_templates():
    """Return all templates from the database"""
    db = SessionLocal()
    try:
        rows: List[Template] = db.query(Template).order_by(Template.created_at.desc()).all()
        return {"templates": [serialize_template(t) for t in rows]}
    finally:
        db.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy"
    details = {
        "status": status,
        "api_key_configured": bool(GEMINI_API_KEY),
        "timestamp": time.time()
    }
    
    # Test Gemini API connectivity if API key is configured
    if GEMINI_API_KEY:
        try:
            # Simple test request to check API connectivity
            url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Hello"}
                        ]
                    }
                ]
            }
            params = {"key": GEMINI_API_KEY}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=headers, params=params)
                if resp.status_code == 200:
                    details["gemini_api"] = "healthy"
                else:
                    details["gemini_api"] = f"error_{resp.status_code}"
                    status = "degraded"
        except Exception as e:
            details["gemini_api"] = f"error: {str(e)}"
            status = "degraded"
    else:
        details["gemini_api"] = "not_configured"
        status = "degraded"
    
    details["status"] = status
    return details

@app.get("/sizes")
async def get_available_sizes():
    """Get all available creative sizes"""
    return {"sizes": AVAILABLE_SIZES}

@app.post("/generate-template/")
async def generate_template(
    prompt: str = Form(...),
    platform: str = Form(...),
    size_name: str = Form(...)
):
    """Generate a new template from scratch using AI"""
    try:
        # Validate platform and size
        if platform not in AVAILABLE_SIZES:
            raise HTTPException(status_code=400, detail=f"Invalid platform. Available platforms: {', '.join(AVAILABLE_SIZES.keys())}")
        
        valid_sizes = [size["name"] for size in AVAILABLE_SIZES[platform]]
        if size_name not in valid_sizes:
            raise HTTPException(status_code=400, detail=f"Invalid size for {platform}. Available sizes: {', '.join(valid_sizes)}")
        
        # Generate template
        template = await generate_template_from_scratch(prompt, size_name, platform)
        
        return {
            "template": template,
            "platform": platform,
            "size": size_name,
            "dimensions": next(size for size in AVAILABLE_SIZES[platform] if size["name"] == size_name)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in generate_template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate template: {str(e)}")


# Chat Context Endpoints
@app.post("/chat/save")
async def save_chat_message(
    session_id: str = Form(...),
    template_id: int = Form(None),
    user_message: str = Form(...),
    ai_response: str = Form(None),
    template_html: str = Form(None)
):
    """Save a chat message to the history"""
    db = SessionLocal()
    try:
        chat_entry = ChatHistory(
            session_id=session_id,
            template_id=template_id,
            user_message=user_message,
            ai_response=ai_response,
            template_html=template_html
        )
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)
        return {"id": chat_entry.id, "message": "Chat message saved successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save chat message: {str(e)}")
    finally:
        db.close()


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    db = SessionLocal()
    try:
        chat_history = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.created_at.asc()).all()
        
        return {
            "session_id": session_id,
            "history": [serialize_chat_history(ch) for ch in chat_history]
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")
    finally:
        db.close()


@app.post("/chat/generate-with-context")
async def generate_with_chat_context(
    session_id: str = Form(...),
    template: str = Form(...),
    prompt: str = Form(...),
    template_id: int = Form(None)
):
    """Generate template with chat context"""
    try:
        # Get chat history for context
        db = SessionLocal()
        try:
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.session_id == session_id
            ).order_by(ChatHistory.created_at.desc()).limit(5).all()
        finally:
            db.close()
        
        # Build context from chat history
        context_messages = []
        for ch in reversed(chat_history):  # Reverse to get chronological order
            if ch.user_message:
                context_messages.append(f"User: {ch.user_message}")
            if ch.ai_response:
                context_messages.append(f"Assistant: {ch.ai_response}")
        
        # Create enhanced prompt with context
        context_text = "\n".join(context_messages) if context_messages else "No previous context."
        enhanced_prompt = f"""Previous conversation context:
{context_text}

Current request: {prompt}

Please consider the previous conversation context when making changes to the template."""
        
        # Generate with enhanced prompt
        html = await call_gemini(template, enhanced_prompt)
        
        # Save this interaction to chat history
        db = SessionLocal()
        try:
            chat_entry = ChatHistory(
                session_id=session_id,
                template_id=template_id,
                user_message=prompt,
                ai_response="Template updated successfully",
                template_html=html
            )
            db.add(chat_entry)
            db.commit()
        finally:
            db.close()
        
        return {
            "template": html,
            "context_used": len(chat_history) > 0,
            "context_messages": len(chat_history)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in generate_with_chat_context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate with context: {str(e)}")


@app.delete("/chat/clear/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    db = SessionLocal()
    try:
        deleted_count = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).delete()
        db.commit()
        return {"message": f"Cleared {deleted_count} chat messages", "session_id": session_id}
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")
    finally:
        db.close()
