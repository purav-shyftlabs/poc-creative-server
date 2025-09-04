import os
import httpx
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import asyncio
import time
import logging
from playwright.async_api import async_playwright
import re
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration
ENABLE_FALLBACK = False  # Set to False to disable fallback HTML generation
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

# Default template
DEFAULT_TEMPLATE = """
<!doctype html>
<!--SIZE: 970x250-->
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pure CSS HTML5 Ad Creative</title>
  <!--
    Pure CSS single-file ad template
    - No frameworks, no external CSS (except optional image assets)
    - Change root class to .size-300x250, .size-728x90, etc.
    - Replace placeholders (logo SVG, product SVG/img, copy).
    - Set window.clickTag (or edit href on the <a class="click">) when served by ad server.
  -->
  <style>
    :root{
      --brand:#4f46e5;
      --accent:#06b6d4;
      --ink:#0b1021;
      --muted:#475569;
      --bg:#f8fafc;
      --cta-color:#ffffff;
      --radius:14px;
      --shadow: 0 8px 24px rgba(2,6,23,0.18);
      --w:300px; --h:250px; /* default */
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Size presets */
    .size-300x250 { --w:300px; --h:250px; }
    .size-336x280 { --w:336px; --h:280px; }
    .size-728x90  { --w:728px; --h:90px;  }
    .size-320x50  { --w:320px; --h:50px;  }
    .size-160x600 { --w:160px; --h:600px; }
    .size-300x600 { --w:300px; --h:600px; }
    .size-970x250 { --w:970px; --h:250px; }

    /* Ad canvas */
    .ad{
      box-sizing:border-box;
      width:var(--w); height:var(--h);
      border-radius:var(--radius);
      overflow:hidden;
      position:relative;
      background: linear-gradient(180deg,#ffffff 0%, #f5f7fb 100%);
      box-shadow:var(--shadow);
      color:var(--ink);
      -webkit-font-smoothing:antialiased;
      -moz-osx-font-smoothing:grayscale;
    }

    /* Decorative blobs */
    .blob{
      position:absolute; border-radius:50%; filter:blur(18px); opacity:.28; pointer-events:none;
    }
    .blob.a{ width:160px; height:160px; right:-40px; top:-40px; background:linear-gradient(135deg,var(--accent),transparent);} 
    .blob.b{ width:140px; height:140px; left:-40px; bottom:-40px; background:linear-gradient(135deg,var(--brand),transparent);} 

    /* Layout grid */
    .content{position:absolute; inset:0; display:grid; grid-template-rows:auto 1fr auto; gap:8px; padding:14px;}

    .top{display:flex; align-items:center; gap:8px;}
    .logo{width:22px; height:22px; border-radius:6px; display:grid; place-items:center; background:linear-gradient(135deg,var(--brand),var(--accent)); box-shadow:0 2px 8px rgba(79,70,229,0.35); flex:0 0 22px;}
    .brand{font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.02em}
    .badge{margin-left:auto; font-size:11px; padding:3px 8px; border-radius:999px; background:rgba(79,70,229,.12); color:#4338ca; font-weight:600}

    .body{display:grid; grid-template-columns:1fr auto; align-items:center; gap:10px}
    .headline{margin:0; font-weight:800; font-size:20px; line-height:1.05}
    .sub{margin:6px 0 0; font-size:12px; color:var(--muted)}

    .product{width:110px; height:110px; border-radius:16px; background:white; box-shadow:0 8px 24px rgba(2,6,23,.18); overflow:hidden; display:flex; align-items:center; justify-content:center; transform:translateZ(0);}

    .cta{display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px; background:linear-gradient(135deg,var(--brand),var(--accent)); color:var(--cta-color); font-weight:700; text-decoration:none; border:0; cursor:pointer; box-shadow:0 6px 18px rgba(6,182,212,.35); transition:transform .18s ease, box-shadow .18s ease}
    .cta:hover{transform:translateY(-2px); box-shadow:0 10px 24px rgba(6,182,212,.45)}
    .cta:focus{outline:3px solid rgba(14,165,233,.18); outline-offset:3px}

    .footer{display:flex; align-items:center; gap:8px; font-size:11px; color:var(--muted)}

    /* Small banner tweeks */
    .size-320x50 .product{display:none}
    .size-320x50 .headline{font-size:14px}
    .size-728x90 .product{width:86px; height:86px}

    /* Animations */
    @keyframes float{0%{transform:translateY(0)}50%{transform:translateY(-6px)}100%{transform:translateY(0)}}
    .product{animation:float 6s ease-in-out infinite}

    /* Clickable overlay (full creative clickable) */
    .click{position:absolute; inset:0; z-index:40; text-indent:-9999px; overflow:hidden}

    /* Accessibility helpers */
    .sr-only{position:absolute!important;height:1px;width:1px;overflow:hidden;clip:rect(1px,1px,1px,1px);white-space:nowrap}

    /* Responsive typography inside different canvas sizes using relative units */
    .ad .headline{font-size:calc(12px + (22 - 12) * ((var(--w) - 160px) / (970 - 160)));}
    .ad .sub{font-size:calc(10px + (12 - 10) * ((var(--w) - 160px) / (970 - 160)))}

  </style>
</head>
<body style="background:transparent; margin:0; padding:0;">

  <!-- Root: change class to one of the size presets -->
  <div class="ad size-970x250" role="region" aria-label="Advertisement">

    <!-- Decorative blobs -->
    <div class="blob a" aria-hidden="true"></div>
    <div class="blob b" aria-hidden="true"></div>

    <div class="content">
      <!-- Top bar -->
      <div class="top">
        <div class="logo" aria-hidden="true">
          <!-- simple svg mark -->
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v6M12 16v6M2 12h6M16 12h6M5 5l4 4M15 15l4 4M5 19l4-4M15 9l4-4"/>
          </svg>
        </div>
        <div class="brand">Your Brand</div>
        <div class="badge">New</div>
      </div>

      <!-- Body -->
      <div class="body">
        <div>
          <h1 class="headline">Level up your workflow</h1>
          <p class="sub">Ship faster with tools designed for speed, clarity, and delight.</p>
          <div style="margin-top:10px;">
            <a class="cta" href="#" onclick="handleClick(event)" role="button" aria-label="Learn more">Learn more
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-left:6px"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </a>
          </div>
        </div>

        <div class="product" aria-hidden="true">
          <!-- Replace with an <img src="..." alt="..."> or inline asset if desired -->
          <svg viewBox="0 0 120 120" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Product preview">
            <defs>
              <linearGradient id="g1" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0%" stop-color="#818cf8"/>
                <stop offset="100%" stop-color="#06b6d4"/>
              </linearGradient>
            </defs>
            <rect x="10" y="22" width="100" height="76" rx="10" fill="url(#g1)" opacity=".18"/>
            <rect x="16" y="28" width="88" height="18" rx="6" fill="#0f172a" opacity=".12"/>
            <rect x="16" y="52" width="72" height="8" rx="4" fill="#0f172a" opacity=".08"/>
            <rect x="16" y="64" width="88" height="8" rx="4" fill="#0f172a" opacity=".08"/>
            <rect x="16" y="76" width="64" height="8" rx="4" fill="#0f172a" opacity=".08"/>
          </svg>
        </div>
      </div>

      <!-- Footer -->
      <div class="footer">
        <div style="display:flex;align-items:center;gap:6px">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg>
          <div style="font-weight:600">4.8</div>
          <div>(2k+)</div>
        </div>
        <div style="margin-left:auto;font-size:11px;color:var(--muted)">* Terms apply</div>
      </div>
    </div>

    <!-- Full clickable layer. Use window.clickTag when served by ad server. -->
    <a class="click" href="#" onclick="handleClick(event)">Open landing page</a>
  </div>

  <script>
    // Click handler supports common ad clickTag pattern.
    function handleClick(e){
      e.preventDefault();
      var href = window.clickTag || e.currentTarget.getAttribute('href') || '#';
      var target = '_blank';
      try{ window.open(href, target); } catch(err){ location.href = href; }
    }

    // Impression beacon helper (set window.impressionUrl before load to use)
    (function fireImpression(){
      var url = window.impressionUrl;
      if(!url) return;
      var img = new Image(); img.src = url + (url.indexOf('?')>-1?'&':'?') + 't=' + Date.now();
    })();

  </script>
</body>
</html>


"""

def render_template(template, variables):
    """Simple template rendering function"""
    rendered = template
    for key, value in variables.items():
        placeholder = "{{" + key + "}}"
        rendered = rendered.replace(placeholder, str(value))
    return rendered

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

def generate_fallback_html(template, prompt):
    """Generate a fallback HTML when Gemini API is unavailable"""
    # Simple fallback - just return the template with a note about the prompt
    fallback_html = template
    
    # Add a simple fallback content based on the prompt
    fallback_html += f"""
    <div class="p-4 bg-white rounded-lg shadow-soft">
        <h1 class="text-2xl font-bold text-gray-800 mb-2">Creative Banner</h1>
        <p class="text-gray-600 mb-4">Generated based on your creative brief</p>
        <p class="text-sm text-gray-500 mb-4">{prompt[:100]}{'...' if len(prompt) > 100 else ''}</p>
        <button class="bg-blue-500 hover:bg-blue-600 text-white font-semibold px-4 py-2 rounded">
            Learn More
        </button>
    </div>
    """
    
    return fallback_html

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
        # "You are a professional HTML email/banner creative generator. "
        # "DO NOT REMOVE ANY COMMENT OR SIZE COMMENT"
        # "IF USER ASKS TO RESIZE OR CHANGE THE POSITIION OF THE ELEMENTS; CHANGE THE POSITION OF THE ELEMENTS ACCORDING TO THE SIZE OF THE TEMPLATE"
        # "IF YOU THINK THERE IS EMPTY SPACE; FILL IT WITH SOMETHING; OF REQUIRED"
        # "Only change what user asks for in the prompt unless you have to update"
        # "Given this HTML template and the provided creative brief, generate a complete HTML banner ad. "
        # "As per the current prompt only, update template"
        # "You can add any tag or remove is user specified. For example, if user says to add a logo, add it."
        # "For image url, check if it is valid and if not, if user give url add it directly"
        # "Images object type should always be 'contain' no matter what user asks for"
        # "IF USER ASKS FOR REDO OR GIVE ME ANOTHER DESIGN; THINK AND BASED ON THE SCENARIO WRITE CODE AND GIVE BACK THAT CODE; but keep the content same and use modern ui design; and completely chanhe the design completely; always instead of images use placeholder; and keep the size of the template same and fit every element according to the size of the template"
        # "Return ONLY the completed HTML, no explanation.\n"
        # f"Template:\n{template}\n"
        # f"Creative Brief:\n{prompt}\n"
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
    "- sometimes user will give wage instruction, you need to think with UI/UX prespective and can change the design. for example, if user says to add a logo, add it. or user says add text under the headline, then check what size of the text will be fit in the space and add it. Think for all the aspects"
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
                logger.error("All retries exhausted for timeout. Using fallback HTML generation.")
                fallback_html = generate_fallback_html(template, prompt)
                return clean_html_output(fallback_html)
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
                    logger.error("All retries exhausted for 503 error. Using fallback HTML generation.")
                    fallback_html = generate_fallback_html(template, prompt)
                    return clean_html_output(fallback_html)
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
                logger.error(f"All retries exhausted. Using fallback HTML generation. Last error: {str(e)}")
                fallback_html = generate_fallback_html(template, prompt)
                return clean_html_output(fallback_html)

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

@app.get("/")
async def root():
    """Root endpoint with default template preview"""
    sample_variables = {
        "headline": "Introducing the Future of Creative Ads!",
        "product": "Gemini-powered Automation, Now Live",
        "cta": "Try Now"
    }
    
    rendered_html = render_template(DEFAULT_TEMPLATE, sample_variables)
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Creative POC - Default Template Preview</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .template-container {{ max-width: 800px; margin: 0 auto; }}
            .code-block {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .preview {{ border: 2px solid #ddd; border-radius: 8px; overflow: hidden; }}
        </style>
    </head>
    <body>
        <div class="template-container">
            <h1 class="text-3xl font-bold mb-6">Creative POC - Default Template Preview</h1>
            
            <h2 class="text-xl font-semibold mb-3">Template Code:</h2>
            <div class="code-block">
                <pre><code>{DEFAULT_TEMPLATE}</code></pre>
            </div>
            
            <h2 class="text-xl font-semibold mb-3">Sample Variables:</h2>
            <div class="code-block">
                <pre><code>{sample_variables}</code></pre>
            </div>
            
            <h2 class="text-xl font-semibold mb-3">Rendered Output:</h2>
            <div class="preview">
                {rendered_html}
            </div>
            
            <div class="mt-8 p-4 bg-blue-50 rounded-lg">
                <h3 class="font-semibold mb-2">API Endpoints:</h3>
                <ul class="list-disc list-inside space-y-1">
                    <li><strong>GET /</strong> - This preview page</li>
                    <li><strong>POST /generate-image/</strong> - Generate image from template and variables</li>
                    <li><strong>GET /preview/</strong> - Preview template with custom variables</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=full_html)

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

@app.get("/test/clean-html")
async def test_clean_html(html_content: str = "```html\n<div>Test</div>\n```"):
    """Test endpoint to verify HTML cleaning functionality"""
    cleaned = clean_html_output(html_content)
    return {
        "original": html_content,
        "cleaned": cleaned,
        "removed_markdown": "```html" in html_content or "```" in html_content,
        "is_valid_html": cleaned.startswith('<') and cleaned.endswith('>')
    }

@app.get("/template/dimensions")
async def get_template_dimensions(template: str = DEFAULT_TEMPLATE):
    """Get the dimensions of a template"""
    width, height = extract_dimensions_from_template(template)
    return {
        "width": width,
        "height": height,
        "size_class": f"size-{width}x{height}"
    }

@app.get("/template/default")
async def get_default_template():
    """Get the default template as JSON"""
    return {
        "template": DEFAULT_TEMPLATE,
        "variables": {
            "headline": "Introducing the Future of Creative Ads!",
            "product": "Gemini-powered Automation, Now Live",
            "cta": "Try Now"
        }
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "enable_fallback": ENABLE_FALLBACK,
        "max_retries": MAX_RETRIES,
        "retry_delay_base": RETRY_DELAY_BASE,
        "rate_limit_interval": min_interval,
        "api_key_configured": bool(GEMINI_API_KEY)
    }

@app.post("/config")
async def update_config(
    enable_fallback: bool = None,
    max_retries: int = None,
    retry_delay_base: float = None,
    rate_limit_interval: float = None
):
    """Update configuration (for debugging/monitoring purposes)"""
    global ENABLE_FALLBACK, MAX_RETRIES, RETRY_DELAY_BASE, min_interval
    
    if enable_fallback is not None:
        ENABLE_FALLBACK = enable_fallback
    if max_retries is not None:
        MAX_RETRIES = max_retries
    if retry_delay_base is not None:
        RETRY_DELAY_BASE = retry_delay_base
    if rate_limit_interval is not None:
        min_interval = rate_limit_interval
    
    return {"message": "Configuration updated", "config": await get_config()}

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
