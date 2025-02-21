from agent import user_proxy, retriever_agent, context_agent, generator_agent, evaluator_agent, query_refiner_agent


def state_transition(last_speaker, groupchat):
    """Controls agent transitions based on workflow logic."""
    if last_speaker is user_proxy:
        print("Transitioning: UserProxy → RetrieverAgent")
        return retriever_agent
    elif last_speaker is retriever_agent:
        print("Transitioning: RetrieverAgent → ContextAgent")
        return context_agent
    elif last_speaker is context_agent:
        print("Transitioning: ContextAgent → GeneratorAgent")
        return generator_agent
    elif last_speaker is generator_agent:
        print("Transitioning: GeneratorAgent → EvaluatorAgent")
        return evaluator_agent
    elif last_speaker is evaluator_agent:
        last_message = groupchat.messages[-1]["content"].strip().lower()

        if "unsatisfactory" in last_message:
            print("Restarting Query Refinement...")
            return query_refiner_agent
        
        print("Transitioning: EvaluatorAgent → End (Satisfactory)")
        return None
    
    elif last_speaker is query_refiner_agent:
        print("Transitioning: QueryRefinerAgent → RetrieverAgent")
        return retriever_agent
