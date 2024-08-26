from utils import get_doc_tools
from pathlib import Path
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
import os
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini-2024-07-18")

llm = OpenAI(temperature=0)

# urls = [
#     "https://openreview.net/pdf?id=VtmBAGCN7o",
#     "https://openreview.net/pdf?id=6PmJoRfdaK",
#     "https://openreview.net/pdf?id=LzPWWPAdY4",
#     "https://openreview.net/pdf?id=VTF8yNQM66",
#     "https://openreview.net/pdf?id=hSyW5go0v8",
#     "https://openreview.net/pdf?id=9WD9KwssyT",
#     "https://openreview.net/pdf?id=yV6fD7LYkF",
#     "https://openreview.net/pdf?id=hnrB5YHoYu",
#     "https://openreview.net/pdf?id=WbWtOYIzIK",
#     "https://openreview.net/pdf?id=c5pwL0Soay",
#     "https://openreview.net/pdf?id=TpD2aG1h0D"
# ]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf",
    "metra.pdf",
    "vr_mcl.pdf"
]
papers = [os.path.join('data', paper) for paper in papers]

# for url, paper in zip(urls, papers):
#     !wget "{url}" -O "{paper}"

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# agent_worker = FunctionCallingAgentWorker.from_tools(
#     all_tools, 
#     llm=llm, 
#     verbose=True
# )
# agent = AgentRunner(agent_worker)

# response = agent.query(
#     "Tell me about the evaluation dataset used in LongLoRA, "
#     "and then tell me about the evaluation results"
# )

# response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
# print(str(response))

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# tools = obj_retriever.retrieve(
#     "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
# )

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
    You are an agent designed to answer queries over a set of given papers.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
    verbose=True
)
agent = AgentRunner(agent_worker)

# response = agent.query(
#     "Tell me about the evaluation dataset used "
#     "in MetaGPT and compare it against SWE-Bench"
# )
# print(str(response))

# response = agent.query(
#     "Compare and contrast the LoRA papers (LongLoRA, LoftQ). "
#     "Analyze the approach in each paper first. "
# )

def main():
    try:
        while True:
            q = input("Enter prompt: ")
            response = agent.query(q)
            # print(str(response), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Goodbye!")

if __name__=="__main__":
    main()