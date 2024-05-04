import os
import json
import boto3
import fitz
import pdf2md
from openai import OpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from upstash_vector import Index, Vector


s3_client = boto3.client("s3")
oai_client = OpenAI()
index = Index.from_env()


def parse_markdown_into_chunks(markdown_text):

    # Split the document based on markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)

    chunks = []

    for i, chunk in enumerate(md_header_splits):
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


def index_pdf(pdf_path):
    doc_id = os.path.basename(pdf_path).split('.')[0]

    # Convert the PDF to markdown
    markdown_text = pdf2md.to_markdown(fitz.open(pdf_path))

    # Load and parse the PDF into chunks
    chunks = parse_markdown_into_chunks(markdown_text)

    # Convert the chunks to vector objects
    vectors = []
    for chunk in chunks:
        chunk['metadata']['doc_id'] = doc_id
        chunk_id = f"{doc_id}_{chunk['id']}"

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
        download_path = "/tmp/{}".format(filename)
        s3_client.download_file(bucket, key, download_path)

        index_pdf(download_path)

        response.append(f"PDF {filename} ingested and indexed")

    return {"statusCode": 200, "body": response}
