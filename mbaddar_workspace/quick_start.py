#
import logging
import os
from langchain.llms import OpenAI  # can be another provider
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    os.environ["OPENAI_API_KEY"] = "..."
    # Longchain is a good LLM-interface
    # https://towardsdatascience.com/the-easiest-way-to-interact-with-language-models-4da158cfb5c5#:~:text=With%20the%20same%20interface%2C%20you,%2C%20and%20self%2Dhosted%20Models.
    # Model providers (open-source and commercial)
    # https://cobusgreyling.medium.com/the-large-language-model-landscape-9da7ee17710b

    # Model init
    llm = OpenAI(temperature=0.9)

    questions = ["What would be a good company name for a company that makes colorful socks?",
                 "How old is the Rock ?",
                 "What is the best place to buy beer?",
                 "How can I fix my leaking sink ?"]
    # quick notes, no clarification questions . No followup ones
    # https://www.indeed.com/career-advice/career-development/clarifying-questions
    # The answer should be more "precise", need to ask clarification questions first
    for question in questions:
        answer = llm(question)
        logger.info(f'question = {question}\n'
                    f'answer = {answer}')

    # Prompts

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    logger.info('Running Chain')
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_res = chain.run("colorful socks")
    logger.info(f'Chain results = {chain_res}')
