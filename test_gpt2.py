import os
import torch
import datasets
import transformers
from transformers import AutoModelForQuestionAnswering
from nn_pruning.patch_coordinator import ModelPatchingCoordinator
from nn_pruning.model_patcher import ModelPatcher
import torch
from nn_pruning.patch_coordinator import SparseTrainingArguments
from transformers.pytorch_utils import Conv1D
def replace_conv1d_with_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            # 获取原始Conv1D层的参数
            weight = module.weight
            bias = module.bias
            
            # 创建一个新的Linear层
            linear_layer = torch.nn.Linear(weight.size(0), weight.size(1), bias=True)
            
            # 复制权重和偏置
            with torch.no_grad():
                linear_layer.weight.copy_(weight.t())  # 注意这里需要转置
                linear_layer.bias.copy_(bias)
            
            # 替换模型中的Conv1D层
            parent_module, child_name = get_parent_module_and_child_name(model, name)
            setattr(parent_module, child_name, linear_layer)

def get_parent_module_and_child_name(model, full_name):
    parts = full_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__} and torch v{torch.__version__}")
model_name = "gpt2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
replace_conv1d_with_linear(model)
sparse_args = SparseTrainingArguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(model)


mpc = ModelPatchingCoordinator(
    sparse_args=sparse_args, 
    device=device, 
    cache_dir="checkpoints", 
    logit_names="logits", 
    model_name_or_path=model_name,
    teacher_constructor=None)

# 对模型进行补丁
patched_model = mpc.patch_model(model)

print(model)
