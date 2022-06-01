from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------- #
# Distributed setup
# To avoid warnings about parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()
# ------------------------------------------------- #


model_name = "lvwerra/codeparrot"
config = AutoConfig.from_pretrained(model_name)
model_hidden_size = 1600


# Batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 3 * world_size

ds_config = {
    "fp16": {
        "enabled": True
    },
    # "bf16": {
    #     "enabled": False
    # },
    "zero_optimization": {
        "stage": 3,
        # "offload_param": {
        #     "device": "none",
        #     "pin_memory": True
        # },
        "overlap_comm": True,
        #"contiguous_gradients": False,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    #"steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 3,
    "wall_clock_breakdown": False
}


# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  
model = AutoModelForCausalLM.from_pretrained(model_name).half()
# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# Deepspeed ZeRO can process unrelated inputs on each GPU. So for 2 gpus you process 2 inputs at once.
# If you use more GPUs adjust for more.
# And of course if you have just one input to process you then need to pass the same string to both gpus
# If you use only one GPU, then you will have only rank 0.
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "def reverse_list(s):"
elif rank == 1:
    text_in = "def shuffle_list(s):"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
print(inputs.size())
inputs = torch.cat([inputs]*3)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
output = [None for _ in text_out]
torch.distributed.all_gather_object(output, text_out[torch.distributed.get_rank()])
print(output)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")