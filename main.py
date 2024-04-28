from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from pdf import canada_engine, lpp_engine, apg_engine, ifd_engine

load_dotenv()

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=lpp_engine,
        metadata=ToolMetadata(
            name="lpp_data",
            description="il s'agit de la loi suisse sur le droit de la prévoyace professionnelle vieillesse, survivants et invalidité", 
        ),
    ),
    QueryEngineTool(
        query_engine=apg_engine,
        metadata=ToolMetadata(
            name="apg_data",
            description="ce document des inforamtions sur les allocations pour perte de gain dans le cadre de l'armée suisse", 
        ),
    ),
    QueryEngineTool(
        query_engine=ifd_engine,
        metadata=ToolMetadata(
            name="ifd_data",
            description="il s'agit de la loi fédérale suisse sur l'impôt fédéral direct", 
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
