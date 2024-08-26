from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import os
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional

Settings.llm = OpenAI(model="gpt-4o-mini-2024-07-18")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

PERSIST_DIR_VECTOR = "./storage/vector_index"
PERSIST_DIR_SUMMARY = "./storage/summary_index"

def load_or_create_index(index_type, persist_dir, nodes=None):
    if not os.path.exists(persist_dir):
        if nodes is None:
            raise ValueError(f"Nodes must be provided to create the {index_type} index.")
        
        if index_type == "vector":
            index = VectorStoreIndex(nodes)
        elif index_type == "summary":
            index = SummaryIndex(nodes)
        else:
            raise ValueError("Invalid index type specified.")
        
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    
    return index


def get_vector_and_summary_index(file_path: str):
    """Get vector index and summary index."""
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    docs_name = file_path.split('/')[-1].split('.')[0]

    summary_index = load_or_create_index("summary", os.path.join(PERSIST_DIR_SUMMARY, docs_name), nodes)
    vector_index = load_or_create_index("vector", os.path.join(PERSIST_DIR_VECTOR, docs_name), nodes)
    
    return vector_index, summary_index


def get_router_query_engine(file_path: str, llm = None, embed_model = None):
    """Get router query engine."""
    vector_index, summary_index = get_vector_and_summary_index(file_path)
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        llm=llm
    )
    vector_query_engine = vector_index.as_query_engine(llm=llm)

    docs_name = file_path.split('/')[-1].split('.')[0]
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {docs_name}"
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            f"Useful for retrieving specific context from the {docs_name} paper."
        ),
    )
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine


def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""
    vector_index, summary_index = get_vector_and_summary_index(file_path)
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over a given paper.
    
        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
        
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool