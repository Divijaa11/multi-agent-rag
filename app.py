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
    agents=[user_proxy, retriever_agent, context_agent, generator_agent, evaluator_agent, query_refiner_agent ],
    messages=[],
    max_round=6,
    speaker_selection_method=state_transition
)

manager = GroupChatManager(groupchat=groupchat)

st.title("Training Tool")

query = st.text_input("Enter your query:")

refinement_limit = 3
refinement_count = 0 

if st.button("Submit"):
    if query:
        user_proxy.initiate_chat(manager, message=query)

        while refinement_count < refinement_limit:
            last_message = groupchat.messages[-1] if groupchat.messages else None
            print("last_message: ", last_message)

            if last_message and last_message["name"] == "QueryRefinerAgent":
                print("Manually re-running with refined query...")
                refined_query = last_message["content"].strip()
                refinement_count += 1
                print(f"Refinement Count: {refinement_count}")
                
                user_proxy.initiate_chat(manager, message=refined_query)
            else:
                break

        if refinement_count >= refinement_limit:
            print("Max refinement attempts reached. Ending refinement.")

        generator_responses = [
            msg["content"] for msg in groupchat.messages if msg["name"] == "GeneratorAgent"
        ]
        response = generator_responses[-1] if generator_responses else "No response generated."
        
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please enter a query.")
