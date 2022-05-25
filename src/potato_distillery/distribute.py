import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

generator = pipeline('text-generation', model='gpt2', device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.half,
                                           replace_method='auto',
                                           replace_with_kernel_inject=False)

string = generator("DeepSpeed is", do_sample=True, min_length=50, max_length=250, num_return_sequences=10)
print(string)