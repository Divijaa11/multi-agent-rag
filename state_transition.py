from agent import user_proxy, retriever_agent, context_agent, generator_agent, evaluator_agent, query_refiner_agent


def state_transition(last_speaker, groupchat):
    if last_speaker is user_proxy:
        return retriever_agent
    elif last_speaker is retriever_agent:
        return context_agent
    elif last_speaker is context_agent:
        return generator_agent
    elif last_speaker is generator_agent:
        return evaluator_agent
    elif last_speaker is evaluator_agent:
        last_message = groupchat.messages[-1]["content"]
        
        if "satisfactory" in last_message.lower():
            return None

        query_refiner_count = sum(1 for msg in groupchat.messages if msg["name"] == "QueryRefinerAgent")
            
        if query_refiner_count >= 3: 
            return None
        
        return query_refiner_agent
    
    elif last_speaker is query_refiner_agent:
        return retriever_agent
