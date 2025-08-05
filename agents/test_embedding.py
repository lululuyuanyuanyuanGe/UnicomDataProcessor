import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.modelRelated import invoke_embedding_model
from dotenv import load_dotenv

load_dotenv()

texts = [
    "耕地地力保护补贴",
    "耕地地力保护补贴",
    "耕地地力保护补贴",
    "耕地地力保护补贴",
    "耕地地力保护补贴",]

embeddings = invoke_embedding_model(model_name="Qwen/Qwen3-Embedding-8B", texts=texts)
for i, embedding in enumerate(embeddings):
    print(f"Text {i+1}: {texts[i]}")
    print(f"Embedding: {embedding}")
    print("-" * 50)






