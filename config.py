from dotenv import load_dotenv
import os
load_dotenv()

class Settings:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.chatbi_address = os.getenv("CHAT_BI_ADDR")
            self.chatbi_port = int(os.getenv("CHAT_BI_PORT", 3306))
            self.chatbi_user = os.getenv("CHAT_BI_USER")
            self.chatbi_password = os.getenv("CHAT_BI_PASSWORD")
            self.chatbi_database = os.getenv("CHAT_BI_DB")
            self.siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
            Settings._initialized = True

settings = Settings()