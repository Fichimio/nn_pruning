from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm
import numpy as np
from evaluate import load
from examples.question_answering.qa_sparse_xp import QASparseXP
# 加载预训练的GPT-2 QA模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = '/home/cmyu/nn_pruning/output_models/squad-gpt2'  # 或者使用其他特定版本如 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name,ignore_mismatched_sizes=True).to(device)
# for name,w in model.named_parameters():
#     print(name,w.shape)
model = QASparseXP.compile_model(model_name).to(device)
for name,w in model.named_parameters():
    print(name,w.shape,w)

# 如果GPT-2没有为QA任务进行微调，则可能需要加载一个针对QA任务进行了微调的模型
# 例如：model = GPT2ForQuestionAnswering.from_pretrained('your-finetuned-model')

# 加载SQuAD 1.1数据集
squad = load_dataset('squad')
tokenizer.pad_token = tokenizer.eos_token
# 定义如何编码数据
# 定义如何编码数据
def encode(examples):
    return tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding='max_length',
        max_length=512,
        return_token_type_ids=True,
        return_attention_mask=True,
    )

# 对数据集进行编码，并且保留'id'字段
encoded_datasets = squad.map(encode, batched=True, remove_columns=['title', 'context', 'question', 'answers'])

# 创建DataLoader
data_collator = default_data_collator
eval_dataloader = DataLoader(encoded_datasets["validation"], collate_fn=data_collator, batch_size=8, shuffle=False)

# 设置模型为评估模式
model.eval()

# 初始化evaluate库中的SQuAD指标计算器
squad_metric = load("squad")

# 开始预测
all_predictions = []
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        # 找到每个样本中最高概率的答案开始和结束位置
        start_positions = torch.argmax(start_logits, dim=-1).cpu().numpy()
        end_positions = torch.argmax(end_logits, dim=-1).cpu().numpy()
        
        # 将答案位置转换成文本形式
        for i, (start, end) in enumerate(zip(start_positions, end_positions)):
            input_ids = batch['input_ids'][i].cpu().numpy()
            answer_text = tokenizer.decode(input_ids[start:end+1])
            all_predictions.append({
                "id": encoded_datasets["validation"][i * len(batch['input_ids']) + i]["id"],
                "prediction_text": answer_text,
            })

# 准备真实答案用于比较
references = [{"id": item["id"], "answers": item["answers"]} for item in squad["validation"]]

# 计算F1和其他指标
results = squad_metric.compute(predictions=all_predictions, references=references)
print(f"F1 Score: {results['f1']}")
print(f"Exact Match: {results['exact_match']}")