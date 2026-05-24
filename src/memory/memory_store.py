from langchain.memory import ConversationBufferMemory

user_memories = {}

def get_memory(email):

    if email not in user_memories:

        user_memories[email] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    return user_memories[email]