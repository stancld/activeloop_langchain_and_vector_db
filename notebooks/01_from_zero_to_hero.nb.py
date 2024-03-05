# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## From Zero to Hero
#
# <hr>
#
# https://learn.activeloop.ai/courses/take/langchain/multimedia/46317643-langchain-101-from-zero-to-hero

# %% [markdown]
# ### LLMs

# %%
from langchain.llms import OpenAI

# %%
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

# %%
llm

# %%
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))

# %% [markdown]
# ### The Chains

# %%
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# %%
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)

# %%
print(chain.run("eco-friendly water bottles"))

# %% [markdown]
# ### The Memory

# %%
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# %%
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

# %%
# Start the conversation
conversation.predict(input="Tell me about yourself.")

# %%
conversation.predict(input="What can you do?")

# %%
conversation.predict(input="How can you help me with data analysis?")

# %%
print(conversation)

# %% [markdown]
# ### Deep Lake VectorStore

# %%
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake

# %%
# instantiate the LLM and embeddings models
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# %%
# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# %%
# create Deep Lake dataset
my_activeloop_org_id = "stancld"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# %%
# add documents to our Deep Lake dataset
db.add_documents(docs)

# %%
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# %%
retrieval_qa

# %%
from langchain.agents import AgentType, Tool, initialize_agent

# %%
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions.",
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# %%
response = agent.run("When was Napoleone born?")
print(response)

# %%
# Should never finish ;]
response = agent.run("When was Lance Armstrong born?")
print(response)

# %%
# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963",
]
db.add_documents(text_splitter.create_documents(texts))

# %%
response = agent.run("When was Michael Jordan born?")
print(response)

# %% [markdown]
# ### Agents in LangChain

# %%
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper

# %%
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# %%
search = GoogleSearchAPIWrapper()

# %%
tools = [
    Tool(
        name="google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events",
    )
]

# %%
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=6,
)

# %%
response = agent("What's Rossum Aurora?")
print(response["output"])

# %% [markdown]
# ### Tools in LangChain

# %%
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper

# %%
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"], template="Write a summary of the following text: {query}"
)

# %%
search = GoogleSearchAPIWrapper()
summarize_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events",
    ),
    Tool(
        name="Summarizer",
        func=summarize_chain.run,
        description="useful for summarizing texts",
    ),
]

# %%
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# %%
response = agent(
    "What's the latest news about the company Rossum? Then please summarize the results."
)
print(response["output"])

# %%
