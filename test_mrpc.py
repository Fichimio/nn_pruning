from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from nn_pruning.inference_model_patcher import optimize_model
import numpy as np
from evaluate import load
import torch
from examples.text_classification.glue_sparse_xp import GlueSparseXP
model_name = "/home/cmyu/nn_pruning/output_models/mrpc-t5-small-sparse"
# model = GlueSparseXP.compile_model(model_name)
# 定义使用的模型和数据集名称
# model_name = "textattack/bert-base-uncased-MRPC"
dataset_name = "glue"
task_name = "mrpc"

# 加载预训练模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = optimize_model(model_name,mode=)
model = GlueSparseXP.compile_model(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # MRPC 是二分类任务
# model = torch.load(model_name)

# 加载GLUE数据集
dataset = load_dataset(dataset_name, task_name)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

# 应用预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 加载评估指标
metric = load("accuracy")

# 定义计算指标的函数
def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    if isinstance(predictions,tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=label_ids)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
    logging_dir='./logs',
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 进行评估
evaluation_results = trainer.evaluate()

# 输出结果
print(f"Accuracy: {evaluation_results['eval_accuracy']:.4f}")