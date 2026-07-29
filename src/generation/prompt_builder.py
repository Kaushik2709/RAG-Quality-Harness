from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

DEFAULT_SYSTEM_TEMPLATE = (
    "You are a helpful, precise RAG assistant. Answer the user's query strictly based on the provided context.\n"
    "If the context does not contain enough information to answer, explicitly state that the information is unavailable.\n"
    "Do not hallucinate or extrapolate beyond the provided text."
)

DEFAULT_HUMAN_TEMPLATE = (
    "Retrieved Context:\n"
    "==================\n"
    "{context}\n\n"
    "==================\n"
    "User Query: {query}\n\n"
    "Answer:"
)

def build_chat_prompt_template(
    system_template: str = DEFAULT_SYSTEM_TEMPLATE,
    human_template: str = DEFAULT_HUMAN_TEMPLATE
) -> ChatPromptTemplate:
    """
    Returns a standard LangChain ChatPromptTemplate instance.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
