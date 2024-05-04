context_prompt = """Retrieved context to answer the query is as follows:
{context_str}
"""

system_prompt = """You are a Q&A bot. You are here to answer questions based on the context retrieved
from a vector index of the chunks of a document. You are prohibited from using prior knowledge and you
can only use the context given. If you need more information, please ask the user. If you cannot answer 
the question from the context, you can tell the user that you cannot answer the question. You can also 
ask for more information from the user. If there is a some remote similarity between context and query
you can say that the context doesn't have that information but here is something that is mentioned. You
have access to the context_retrieval tool which you can use multiple times to query for different things
based on the user query and get the relevant context and information to better answer the query."""


system_message = {"role": "system", "content": system_prompt}


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "context_retrieval",
            "description": "This function let's you semantically retrieve relevant context chunks from a given document \
                            based on a query. Based on the original user query, write a good search query which is more \
                            logically sound to retrieve the relevant information from the document. You might even have \
                            to break down the user query into multiple search queries and call this function multiple \
                            times separately if there are multiple questions being asked in the original user query. \
                            This function returns the retrieved context chunks from the document based on the search \
                            query formatted as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The sub-query to search for in the document.",
                    }
                },
                "required": ["search_query"],
            },
        },
    }
]


def build_result_str(metadata):
    """
    Build a string from the metadata dictionary of a query result for adding to the context of the LLM.
    """
    text = metadata["text"]
    _meta = {k: v for k, v in metadata.items() if k != "text"}

    meta_str = "\n".join([f"{k}: {v}" for k, v in _meta.items()])
    return f"{meta_str}\n\n{text}"


def build_context_prompt(retrieval_results):
    context_str = "\n\n---------------------\n\n".join(
        [build_result_str(r.metadata) for r in retrieval_results]
    )
    return context_prompt.format(context_str=context_str)
