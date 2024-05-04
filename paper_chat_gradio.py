import os
import requests
import requests.exceptions
import boto3
import gradio as gr


CHAT_LAMBDA_URL = (
    "https://wvqncbbfwquz3jevyofhrqtvpe0gsmfw.lambda-url.us-east-1.on.aws/"
)

s3_client = boto3.client("s3")
BUCKET_NAME = "project-ava"
S3_PREFIX = "arxiv_pdfs"


def upload_to_s3(filepath):
    s3_key = f"{S3_PREFIX}/{os.path.basename(filepath)}"
    s3_client.upload_file(filepath, BUCKET_NAME, s3_key)


def upload_file(files):
    file_paths = [file.name for file in files]

    for filepath in file_paths:
        upload_to_s3(filepath)

    gr.Info(
        "PDF files uploaded and getting Indexed in the background. Kindly wait for a minute or two before chatting."
    )


def chat(
    message,
    chat_id=None,
):
    body = {"query": message, "chat_id": chat_id}
    response = requests.post(CHAT_LAMBDA_URL, json=body)
    if response.status_code == 200:
        response_body = response.json()
        print(response_body)

        for tool_call in response_body['tool_calls']:
            gr.Info(f"Called Function: `{tool_call}`")

        return response_body["response"], response_body["chat_id"]
    else:
        error_log = f"Status Code: {response.status_code}\nResponse: {response.text}"
        print(error_log)
        gr.Error(error_log)


with gr.Blocks() as demo:

    # Maintain session state here
    chat_id = gr.State(None)
    status_message = gr.State("")

    gr.Markdown("# PaperChat")
    gr.Markdown(
        "<p>This is a simple frontend interface to interact with PaperChat, a simple chatbot to ask questions about the arXiv research papers you upload.</p>"
    )

    # Document Uploads
    with gr.Row():
        with gr.Column(scale=1):
            upload_btn = gr.UploadButton(
                label="Upload the PDF of the arXiv paper",
                file_types=["pdf"],
                file_count="multiple",
                variant="primary"
            )

            upload_btn.upload(upload_file, inputs=upload_btn)

    # Chat Interface
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="PaperChat")
            user_message = gr.Textbox(label="User Message")

            clear = gr.ClearButton(
                [user_message, chatbot, chat_id], value="New Chat", size="sm"
            )

            def respond(message, chat_history, chat_id):
                print("Chat ID:", chat_id)

                response, chat_id = chat(message, chat_id)
                chat_history.append((message, response))

                return "", chat_history, chat_id

            user_message.submit(
                respond,
                inputs=[user_message, chatbot, chat_id],
                outputs=[user_message, chatbot, chat_id],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
