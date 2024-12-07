from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from .retrieve_tool import Retrieve
from prompts import system_prompt

def llm_model(model, temperature, top_p, max_tokens, gemini_api) -> ChatGoogleGenerativeAI :
   return ChatGoogleGenerativeAI(
      model=model,
      temperature=temperature,
      top_p=top_p,
      max_tokens=max_tokens,
      api_key=gemini_api
   )

def retrieve_or_respon(state: MessagesState, Retrieve: Retrieve, vector_store):
   """Tentukan langkah untuk menggunakan tools atau langsung merespon"""
   llm_with_tools = llm_model.bind_tools([Retrieve(vector_store=vector_store)])
   response = llm_with_tools.invoke(state['messages'])
   return {"messages": [response]}

def generate(state: MessagesState):
   """Hasilkan respon"""
   recent_tool_messages = []
   for message in reversed(state["messages"]):
      if message.type == "tool":
         recent_tool_messages.append(message)
      else:
         break
   tool_messages = recent_tool_messages[::-1]

   docs_content = "\n\n".join(doc.content for doc in tool_messages)
   system_message_content = SystemMessage(content=system_prompt(docs_content))
   conversation_messages = [
      message
      for message in state["messages"]
      if message.type in ("human", "system")
      or (message.type == "ai" and not message.tool_calls)
   ]
   prompt = [SystemMessage(system_message_content)] + conversation_messages

   response = llm_model.invoke(prompt)
   return {"messages": [response]}

def graph_architecture(Retrieve: Retrieve, vector_store) -> StateGraph:
   graph_builder = StateGraph(MessagesState)
   tools = ToolNode([Retrieve(vector_store=vector_store)])
   memory = MemorySaver()

   graph_builder.add_node(retrieve_or_respon)
   graph_builder.add_node(tools)
   graph_builder.add_node(generate)

   graph_builder.set_entry_point("retrieve_or_respon")
   graph_builder.add_conditional_edges(
      "retrieve_or_respon",
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