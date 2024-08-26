from utils import get_router_query_engine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-4o-mini-2024-07-18")
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

query_engine = get_router_query_engine("data/metagpt.pdf", llm=llm, embed_model=embed_model)

def main():
    try:
        while True:
            q = input("Enter prompt: ")
            response = query_engine.query(q)
            print(str(response), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Goodbye!")

if __name__=="__main__":
    main()