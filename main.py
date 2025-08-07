from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from LLM_model import *

app = FastAPI()


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# @app.get("/", response_class=HTMLResponse)
# async def serve_home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "title": "FastAPI Chatbot"})

API_KEY = os.getenv("SECRET_KEY")

@app.get("/")
def read_root():
    return {"message": "InsureAI is running!"}

# Request schema
class QARequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
@app.post("/hackrx/run")
async def run_qa(
    payload: QARequest,
    authorization: Optional[str] = Header(None)
):
    # Validate token
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    try:
        # Modular processing
        print("run_qa_on_pdf")
        answers = run_qa_on_pdf(str(payload.documents), payload.questions)

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn

#     port = int(os.environ.get("PORT", 8000))  # default to 8000 if not set
#     # uvicorn.run("main:app", host="0.0.0.0", port=port)
