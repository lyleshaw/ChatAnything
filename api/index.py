import uuid

import openai
from fastapi import FastAPI, UploadFile, File, Body
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates

from api.utils import handlePDF

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get("/")
async def index():
    return Jinja2Templates(directory="public").TemplateResponse("index.html", {"request": {}})

@app.post("/upload_file")
async def create_upload_file(file: UploadFile = File(...)):
    file_uuid = str(uuid.uuid4())
    file_location = f"uploads/{file_uuid}.pdf"
    with open(file_location, "wb") as file_object:
        file_object.write(file.file.read())
    handlePDF.extract_pdf(f"embeddings/{file_uuid}.pkl", f"uploads/{file_uuid}.pdf")
    return {
        "code": 1000,
        "msg": "success",
        "data": {
            "file_uuid": file_uuid
        }
    }


@app.post("/ask")
async def ask_question(question: str = Body(), file_uuid: str = Body()):
    answer, context = handlePDF.ask(f"embeddings/{file_uuid}.pkl", question)

    return {
        "code": 1000,
        "msg": "success",
        "data": {
            "answer": answer,
            "ref": context
        }
    }

if __name__ == '__main__':
    openai.api_key = "sk-"
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
