import os
import json
import pickle
import uuid
import boto3
from openai import OpenAI
from upstash_vector import Index
from prompt_utils import system_message, build_context_prompt, tools_schema


s3_client = boto3.client("s3")
BUCKET_NAME = "project-ava"
S3_PREFIX = "paper_chat"

oai_client = OpenAI()
index = Index.from_env()


def context_retrieval(search_query: str) -> str:
    """
    This function let's you semantically retrieve relevant context chunks from a given document based on a query.

    Arguments:
        query (str): The query to search for in the document. Based on the original user query, write a good search query
                     which is more logically sound to retrieve the relevant information from the document.

    Returns:
        str: The retrieved context chunks from the document based on the search query formatted as a string.
    """
    # Get the embeddings for the search query
    query_vector = (
        oai_client.embeddings.create(input=search_query, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

    # Execute the query
    query_result = index.query(
        vector=query_vector, include_metadata=True, include_vectors=False, top_k=3
    )

    return build_context_prompt(query_result)


# A mapping of the tool name to the function that should be called
available_functions = {
    "context_retrieval": context_retrieval,
}


def conversation_turn(
    user_message,
    messages,
    tools,
    model="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=512,
    verbose=True,
    **kwargs,
):

    # Add user message to messages list
    messages.append({"role": "user", "content": user_message})

    if verbose:
        print("\n<< User Message >>")
        print(user_message)

    # Send the conversation and available tools/functions to the model
    response = oai_client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Add the response to the messages list
    messages.append(response_message)

    # Check if the model wanted to call a function
    if tool_calls:

        # Call each of the functions
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(
                    f"\n<< Calling Function `{function_name}` with Args: {function_args} >>"
                )

            # Call the function
            function_response = function_to_call(**function_args)

            if verbose:
                print("<< Function Response >>")
                print(function_response)

            # Add the function response to the messages list
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        # Get a new response from the model based on the function response
        second_response = oai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        second_response_message = second_response.choices[0].message
        messages.append(second_response_message)

        if verbose:
            print("\n<< Response >>")
            print(second_response_message.content)

        # return tool_calls for debugging
        tool_calls_json = [
            {tool.function.name: json.loads(tool.function.arguments)}
            for tool in tool_calls
        ]

        return second_response_message, messages, tool_calls_json

    if verbose:
        print("\n<< Response >>")
        print(response_message.content)

    return response_message, messages, []


def save_chat(chat_id, memory):

    s3_client.put_object(
        Body=pickle.dumps(memory),
        Bucket=BUCKET_NAME,
        Key=f"{S3_PREFIX}/{chat_id}",
    )


def load_chat(chat_id):
    response = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"{S3_PREFIX}/{chat_id}",
    )
    return pickle.loads(response["Body"].read())


def lambda_handler(event, context):
    print(event)
    body = (
        json.loads(event.get("body"))
        if isinstance(event.get("body"), str)
        else event.get("body")
    )

    query = body["query"]
    chat_id = body.get("chat_id", None)
    doc_ids = body.get("doc_ids", [])  # TODO document id based metadata filtering

    # Create a new chat if chat_id is None
    if chat_id is None:
        chat_id = f"chat_{str(uuid.uuid4())}"
        messages = [system_message]
    else:
        messages = load_chat(chat_id)

    response, messages, tool_calls_json = conversation_turn(
        query, messages, tools_schema
    )

    # Save chat
    save_chat(chat_id, messages)

    reponse_body = {
        "response": response.content,
        "chat_id": chat_id,
        "tool_calls": tool_calls_json,
    }

    return {"statusCode": 200, "body": reponse_body}
