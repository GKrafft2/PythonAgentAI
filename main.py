from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_path="models/Meta-Llama-3-8B-Instruct.Q4_1.gguf",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


""" # will print the response once its done generating
response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
print(response.text) """

# prints while the response is beeing generated
response_iter = llm.stream_complete("translate the following text to french: Die Spanngurten der Treppen abnehmen und in der grauen Kiste versorgen")
for response in response_iter:
    print(response.delta, end="", flush=True)
    if response == completion_to_prompt:
        break
