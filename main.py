import os
import uuid
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Redis
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import HumanMessage

# set your openAI api key as an environment variable
os.environ['OPENAI_API_KEY'] = ""

embeddings = OpenAIEmbeddings()

schema = {
    "source": "TEXT",                 # Text field for the source
    "content_vector": "VECTOR",       # Vector field for embeddings
    "content": "TEXT"                 # Text field for content
}

#Enter Redis cluster details redis_url, password, index_name, schema
rds = Redis.from_existing_index(
    embeddings, redis_url="XXXXXX", password='XXXXXX', index_name="chunk", schema=schema
)

retriever = rds.as_retriever(search_kwargs={'k': 3, 'score_treshold': 0.9}, search_type="similarity")

model = ChatOpenAI(
    temperature= 0,
    model_name= 'gpt-4o',
  )


tool = create_retriever_tool(
    retriever,
    name="search_person",
    description="This is where we query the Redis vector database",
)


memory = MemorySaver()
app = create_react_agent(
    model,
    [tool],
    checkpointer=memory,
)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}



input_message = HumanMessage(content="Who is Robert blue")

# user prompt
#prompt_template = PromptTemplate.from_template("{user_message}")
#formatted_prompt = prompt_template.format(user_message=input_message.content)


for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Final output only without trace
#ai_message = None  # Initialize to store the final message
#for event in app.stream({"messages": [formatted_prompt]}, config, stream_mode="values"):
#    ai_message = event["messages"][-1].content  # Extract the AI's content

# Print only the final AI message
#if ai_message:
#    print(ai_message)

