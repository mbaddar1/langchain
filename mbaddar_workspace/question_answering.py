# ref https://python.langchain.com/en/latest/use_cases/question_answering.html
import logging
import os

from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "..."
    ##
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    loader = TextLoader('/home/mbaddar/Downloads/sample_tedata_help.txt')
    logger.info(f'Text Loaded !')
    query = "How can I fix my router?"
    index = VectorstoreIndexCreator().from_loaders([loader])
    res = index.query_with_sources(query)
    print(res)


