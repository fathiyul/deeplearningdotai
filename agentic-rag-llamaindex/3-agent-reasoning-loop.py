from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from utils import get_doc_tools
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini-2024-07-18")

llm = OpenAI(temperature=0)

file_path = "data/metagpt.pdf"
vector_tool, summary_tool = get_doc_tools(file_path, "metagpt")

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)


def main():
    try:
        while True:
            q = input("Enter prompt: ")
            response = agent.chat(q)
            print(10*"-#-"+"Summarizing the answer...\n", str(response), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Goodbye!")

if __name__=="__main__":
    main()