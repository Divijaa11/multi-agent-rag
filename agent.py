from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from config import LLM_CONFIG,CONFIG_LIST

user_proxy = UserProxyAgent(
    name="UserProxy",
    description="Represents the user's input and initiates the chat.",
    system_message=("You are the user's representative and the primary point of interaction in a multi-agent system. "
        "Your role is to:\n"
        "1. Accept and understand the user's queries.\n"
        "2. Clearly communicate these queries to other agents in the system.\n"),
    code_execution_config={
        "use_docker": False,
    },
    human_input_mode="NEVER",
)

retriever_agent = AssistantAgent(
    name="RetrieverAgent",
    description="Fetches the most relevant contexts from ChromaDB.",
    system_message="Use the function `retrieve_contexts` to retrieve relevant contexts from the database.",
    llm_config={
        "config_list": CONFIG_LIST,
        "tools": [ 
            {
                "type": "function", 
                "function": {
                    "name": "retrieve_contexts",
                    "description": "Fetch relevant contexts from the database based on a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The user's search query"},
                        },
                        "required": ["query"],
                    },
                }
            }
        ]
    }
)

context_agent = AssistantAgent(
    name="ContextAgent",
    description="Gathers Context",
    system_message=(
        "Gather the context"
    ),
    llm_config=LLM_CONFIG
)

generator_agent = AssistantAgent(
    name="GeneratorAgent",
    description="Generates beginner-friendly responses with examples, historical background, and follow-up questions, including references to source documents.",
    system_message=(
        '''You are a response generator. Your task is to process the provided context and create a clear, structured, and beginner-friendly response. Include references to the source documents for each piece of information you provide. Follow this structure:

        1. **Direct Answer**:
           - Extract and clearly state the most relevant answer to the query from the context.
           - Ensure the answer is concise and directly addresses the query.

        2. **Examples**:
           - List and summarize all relevant examples or datasets from the context.
           - Include key details such as numbers, trends, or comparisons for each example.
           - For each example, include a reference to the source document.

        3. **Historical Background**:
           - Provide a brief explanation of how this topic has evolved over time or what factors influence it.
           - Mention relevant historical events, changes in trends, or technological advancements if applicable.
           - Include references to the documents that support this information.

        4. **Follow-Up Questions**:
           - Suggest 2-3 related questions to encourage further exploration or discussion.

        ### Example Output Format:
        #### Answer:
        Provide the direct answer to the query based on the context.

        #### Examples:
        - Example 1: Key details. **[Source: Document Name, Page Number]**
        - Example 2: Key details. **[Source: Document Name, Page Number]**
        - Example 3: Key details. **[Source: Document Name, Page Number]**

        #### Historical Background:
        Provide relevant background information or trends. **[Source: Document Name or URL, Page Number]**

        #### Follow-Up Questions:
        - Question 1
        - Question 2
        - Question 3

        ### Additional Guidelines:
        - Use only the provided context to generate your response. Do not invent information.
        - If the context does not contain enough information, state, "The context does not provide sufficient details to answer this query."
        - Clearly cite the source documents for all examples and background information.
        - Avoid technical jargon unless it is defined or explained in simple terms.
        - Keep your tone simple, clear, and accessible to beginners.'''
    ),
    llm_config=LLM_CONFIG
)

evaluator_agent = AssistantAgent(
    name="EvaluatorAgent",
    description="Evaluates the generated response for clarity, quality, and alignment with beginner-level understanding.",
    system_message=(
        "You are a response evaluator. Review the response for:\n"
        "- Clarity: Is it beginner-friendly?\n"
        "- Relevance: Does it address the query?\n"
        "- Completeness: Does it include background, examples, and follow-up questions?\n"
        "Provide constructive feedback if improvements are needed.\n"
        "At the end of your evaluation, explicitly state whether the response is 'satisfactory' or 'unsatisfactory'.\n"
        "If the response is unsatisfactory, provide specific feedback on what needs to be improved.\n"
        "Include the original query in your feedback."
    ),
    llm_config=LLM_CONFIG
)

query_refiner_agent = AssistantAgent(
    name="QueryRefinerAgent",
    description="Refines the query based on feedback from the EvaluatorAgent.",
    system_message=(
        "You are a query refiner. Your role is to refine the user's query based on feedback from the EvaluatorAgent.\n"
        "1. Carefully read the feedback from the EvaluatorAgent.\n"
        "2. Identify the original query and the areas for improvement.\n"
        "3. Modify the query to make it more specific, clear, or relevant.\n"
        "4. Return the refined query."
    ),
    llm_config=LLM_CONFIG
)
