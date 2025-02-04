from autogen import register_function,GroupChat,GroupChatManager
from retriever import retrieve_contexts_with_relevance
import streamlit as st
from agent import user_proxy,retriever_agent, context_agent, generator_agent, evaluator_agent,query_refiner_agent
from state_transition import state_transition

register_function(
    retrieve_contexts_with_relevance,
    caller=retriever_agent,
    executor=context_agent,
    name="retrieve_contexts",
    description="Fetch relevant contexts from the database based on a query."
)

groupchat = GroupChat(
    agents=[user_proxy, retriever_agent,context_agent, generator_agent, evaluator_agent, query_refiner_agent ],
    messages=[],
    max_round=6,
    speaker_selection_method=state_transition
)

manager = GroupChatManager(groupchat=groupchat)

st.title("Training Tool")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    groupchat.messages.clear()
    user_proxy.initiate_chat(manager, message=query)

    generator_responses = [msg["content"] for msg in groupchat.messages if msg["name"] == "GeneratorAgent"]
    response = generator_responses[-1] if generator_responses else "No response generated."
    
    st.subheader("Response:")
    st.write(response)