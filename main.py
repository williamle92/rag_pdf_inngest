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
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

@inggest_client.create_function(
    fn_id="Ingest PDF",
    trigger=inngest.TriggerEvent(event="ingest_pdf"),
)
async def ingest_pdf(ctx: inngest.Context) -> None: 
    return {"hello": "world"}


app: FastAPI = FastAPI()

serve(app, inggest_client, [ingest_pdf])
