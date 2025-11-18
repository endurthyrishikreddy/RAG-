import uuid
from collections import defaultdict , deque


class MemoryManager:
    def __init__(self, short_term_limit=5):
        self.short_term_memory = defaultdict(
            lambda: deque(maxlen=short_term_limit)
        )  # Stores recent interactions
        # Stores all interactions by session_id
        pass

    def new_session(self):
        session_id = str(uuid.uuid4())
        # Init empty deque for session
        self.short_term_memory[session_id] = deque(maxlen=5)
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        message = {"role": role, "content": content}
        self.short_term_memory[session_id].append(message)

    def get_short_term_memory(self, session_id: str):
        return list(self.short_term_memory[session_id])
       