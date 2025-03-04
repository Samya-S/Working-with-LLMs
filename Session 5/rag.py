# importing all necessary libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage

# setting up the openai api key in the os envoirment
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"

class RAG:
    def __init__(self, file_path):
        self.file_path = file_path

    def output(self, questions):
        documents = SimpleDirectoryReader(self.file_path).load_data()
        index = VectorStoreIndex(documents, show_progress=True)

        # we are going to store the data into local disk
        index.storage_context.persist(persist_dir="D:/GitHub/Working with LLMs/Session 5/storage/cache/resume/sleep")

        # reading the embeddings from the local disk
        storage_context = StorageContext.from_defaults(persist_dir="D:/GitHub/Working with LLMs/Session 5/storage/cache/resume/sleep")
        index = load_index_from_storage(storage_context)

        query_engine = index.as_query_engine()

        result = query_engine.query(questions)
        return result.response
    
obj = RAG("D:/GitHub/Working with LLMs/Session 5/data")
print(obj.output("List down all the skills of all candidates"))