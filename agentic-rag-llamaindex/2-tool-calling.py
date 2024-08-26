from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from utils import get_doc_tools
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini-2024-07-18")

llm = OpenAI(temperature=0)

# def add(x: int, y: int) -> int:
#     """Adds two integers together."""
#     return x + y

# def mystery(x: int, y: int) -> int: 
#     """Mystery function that operates on top of two numbers."""
#     return (x + y) * (x + y)

# add_tool = FunctionTool.from_defaults(fn=add)
# mystery_tool = FunctionTool.from_defaults(fn=mystery)


# response = llm.predict_and_call(
#     [add_tool, mystery_tool], 
#     "Tell me the output of the mystery function on 2 and 9", 
#     verbose=True
# )
# print(str(response))

file_path = "data/metagpt.pdf"
vector_query_tool, summary_tool = get_doc_tools(file_path, "metagpt")


def main():
    try:
        while True:
            q = input("Enter prompt: ")
            response = llm.predict_and_call(
                [vector_query_tool, summary_tool], 
                q, 
                verbose=True
            )
            for n in response.source_nodes:
                print(n.metadata)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Goodbye!")

if __name__=="__main__":
    main()