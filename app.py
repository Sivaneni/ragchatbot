from upstash_vector import Vector
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever,LineListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def lambda_handler(event, context):
    # TODO implement
        print(event)
    # Initialize Upstash Vector Store

    # Initialize the embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        store = UpstashVectorStore(embedding=embeddings,index_url="https://present-panther-47615-us1-vector.upstash.io",index_token="ABkFMHByZXNlbnQtcGFudGhlci00NzYxNS11czFhZG1pblpERmhOVFEzTnpNdE1EaG1PQzAwTTJOaUxXRTFZelV0Tm1VeFlUSXhZelJrTVdRdw==")


        llm = OpenAI()
        output_parser = LineListOutputParser()
        QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""Generate exact response from the question{question}"""
            )
        llm_chain = QUERY_PROMPT | llm | output_parser
        retriever = MultiQueryRetriever(
                retriever=store.as_retriever(), llm_chain=llm_chain, parser_key="lines"
            )
        qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
        query = event['query']
        response = qa_chain.run(query)
            
        return {
                'statusCode': 200,
                'body': response
            }
            