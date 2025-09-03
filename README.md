# Creative POC Backend

A FastAPI backend service that generates creative images using Gemini AI and Puppeteer.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Gemini API key:**
   You need a Google Gemini API key. Get one from [Google AI Studio](https://makersuite.google.com/app/apikey).

   Then set the environment variable:
   ```bash
   export GEMINI_API_KEY="your_actual_api_key_here"
   ```

   Or create a `.env` file in the backend-poc directory:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

- `POST /generate-image/` - Generate an image from HTML template and variables
  - Form parameters:
    - `template`: HTML template with placeholders
    - `headline`: Headline text
    - `product`: Product name
    - `cta`: Call-to-action text

## Troubleshooting

- **403 Forbidden Error**: Make sure your `GEMINI_API_KEY` is set correctly
- **Missing API Key**: The server will show a warning if the API key is not configured
- **Browser Issues**: Make sure you have Chrome/Chromium installed for Puppeteer
