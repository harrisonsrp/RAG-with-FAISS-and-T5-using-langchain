# Ingestion libraries
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from LLM.models import t5_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Retrieve libraries
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

if __name__ == '__main__':
    loader = TextLoader("data.txt")
    document = loader.load()
    llm = t5_model().load_model()
    text_splitter = CharacterTextSplitter(
        chunk_size=1500, chunk_overlap=0, separator="\n"
    )
    texts = text_splitter.split_documents(document)
    embeddings = HuggingFaceEmbeddings(model_name="t5-base")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_embed")
    new_vector = FAISS.load_local("faiss_embed", embeddings, allow_dangerous_deserialization=True)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    query = {"input": "who is elon musk?"}
    result = retrival_chain.invoke(input=query)
    print(result['answer'])
