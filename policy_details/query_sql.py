import os
from flask import Flask, jsonify, request

from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                     openai_api_key=os.environ['OPEN_API_KEY'])

def sql_template(llm, question):
    template = """ 
        Given an input question, first create a syntactically correct sql query to run if the user wants to return the numerical value,  
        then look at the results of the query and return the answer. If the user specifies in the question he wants a textual value to obtain return -1. If the query does not provide enough information then return -1. Separate results by comma. Round float numbers to integers.
        The question: {question}
        """
    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    db = SQLDatabase.from_uri("sqlite:///policies.db")
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, top_k=1)

    result = db_chain.run(template.format(question=question))
    return result

@app.route("/api/prompt_sql_route", methods=["POST"])
def prompt_sql_route():
    user_prompt = request.form.get("user_prompt")
    if user_prompt:
        # Get the answer from the chain
        answer = sql_template(llm, user_prompt)

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
        }
        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400

def qa_template(llm, question):
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    return result["result"]

def main():
    # loader = PDFMinerLoader("../SOURCE_DOCUMENTS/Motor_insurance_optimum.pdf")
    # data = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    # all_splits = text_splitter.split_documents(data)

    # EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
    # DEVICE_TYPE = "cpu"
    # # EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
    #
    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(
    #     openai_api_key='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')
    #                                     # EMBEDDINGS
    #                                     )
    question = "How many people obtained the motor insurance in the last 3 months with the total premium exceeding 3K euros"

    sql_answer = sql_template(llm, question=question)
    print(sql_answer)

    answer = qa_template(llm, question)
    print(answer)

if __name__ == "__main__":
    app.run(debug=False, port=5120)
