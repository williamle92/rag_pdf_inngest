import logging
from fastapi import FastAPI
import inngest
from dotenv import load_dotenv
import uuid
import os
import datetime

from inngest.fast_api import serve

load_dotenv()

inggest_client: inngest.Inngest = inngest.Inngest(
    app_id="rag-chat-bot",
    logging=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

app: FastAPI = FastAPI()

serve(app, inggest_client, [])
