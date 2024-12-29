import os
import json
import boto3
import fitz
import pdf2md
from openai import OpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from upstash_vector import Index, Vector
from langchain.document_loaders import JSONLoader,S3FileLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
s3_client = boto3.client("s3")
oai_client = OpenAI()
index = Index.from_env()


def parse_markdown_into_chunks(documents):

    # Split the document based on markdown headers
    #headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),]

    #markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    #md_header_splits = markdown_splitter.split_text(markdown_text)

    # Split the md sections further based on tokens
    chunk_size = 300
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splits = text_splitter.split_documents(documents)

    chunks = []

    for i, chunk in enumerate(splits):
        response = oai_client.embeddings.create(
            input=chunk.page_content, model="text-embedding-3-small"
        )

        _chunk = {
            "id": i,
            "embedding": response.data[0].embedding,
            "metadata": {"text": chunk.page_content},
        }
        chunks.append(_chunk)

    return chunks


def index_data(filename):
    loader = S3FileLoader("agodahotelinfo", filename)
    docs=loader.load()
    documents = []
    for doc in docs:
         documents.append(doc)
    
    # Convert the PDF to markdown
    # markdown_text = pdf2md.to_markdown(fitz.open(pdf_path))

    # Load and parse the PDF into chunks
    chunks = parse_markdown_into_chunks(documents)

    # Convert the chunks to vector objects
    vectors = []
    for chunk in chunks:
        chunk["metadata"]["doc_id"] = filename
        chunk_id = f"{filename}_{chunk['id']}"

        vector = Vector(
            id=chunk_id, vector=chunk["embedding"], metadata=chunk["metadata"]
        )
        vectors.append(vector)

    # Upsert the vectors to the index
    index.upsert(vectors)


def lambda_handler(event, context):
    print(event)

    response = []

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        filename = key.split("/")[-1]
        #download_path = "/tmp/{}".format(filename)
        #s3_client.download_file(bucket, key, download_path)


        index_data(filename)

        response.append(f"data for  {filename} ingested and indexed")

    return {"statusCode": 200, "body": response}
