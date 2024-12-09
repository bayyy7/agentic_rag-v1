from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from .retrieve_tool import Retrieve
from prompts.system_prompt import system_prompt
from config.config import Configuration
from dotenv import load_dotenv
import os

load_dotenv()
config = Configuration().qna_model
gemini_api = os.environ["GOOGLE_GENERATIVE_AI"]
llm_model = ChatGoogleGenerativeAI(
   model=config.model,
   temperature=config.temperature,
   top_p=config.top_p,
   max_tokens=config.max_tokens,
   api_key=gemini_api
)

def retrieve_or_response(state: MessagesState, vector_store, search_type, k):
   """Tentukan langkah untuk menggunakan tools atau langsung merespon"""
   llm_with_tools = llm_model.bind_tools([Retrieve(vector_store=vector_store, search_type=search_type, k=k)])
   response = llm_with_tools.invoke(state['messages'])
   return {"messages": [response]}

def generate(state: MessagesState):
   """Generate a response."""
   print(state["messages"])
   recent_tool_messages = []
   for message in reversed(state["messages"]):
      if message.type == "tool":
         recent_tool_messages.append(message)
      else:
         break
   tool_messages = recent_tool_messages[::-1]
   system_message_content = system_prompt(tool_messages)

   conversation_messages = [
      message
      for message in state["messages"]
      if message.type in ("human", "system")
      or (message.type == "ai" and not message.tool_calls)
   ]
   prompt = [SystemMessage(system_message_content)] + conversation_messages
   print(prompt)
   response = llm_model.invoke(prompt)
   return {"messages": [response]}

def graph_architecture(vector_store, search_type, k) -> StateGraph:
   graph_builder = StateGraph(MessagesState)
   tools = ToolNode([Retrieve(vector_store=vector_store, search_type=search_type, k=k)])
   memory = MemorySaver()

   graph_builder.add_node("retrieve_or_response", lambda MessagesState: retrieve_or_response(MessagesState, vector_store, search_type, k))
   graph_builder.add_node(tools)
   graph_builder.add_node("generate", generate)

   graph_builder.set_entry_point("retrieve_or_response")
   graph_builder.add_conditional_edges(
      "retrieve_or_response",
      tools_condition,
      {
         END: END,
         "tools": "tools"
      }
   )

   graph_builder.add_edge(start_key="tools", end_key="generate")
   graph_builder.add_edge(start_key="generate", end_key=END)
   graph = graph_builder.compile(checkpointer=memory)

   graph_builder.add_edge(start_key="tools", end_key="generate")
   graph_builder.add_edge(start_key="generate", end_key=END)
   graph = graph_builder.compile(checkpointer=memory)
   return graph