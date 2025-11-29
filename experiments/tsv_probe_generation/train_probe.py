import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


class HallucinationProbe(nn.Module):
    """简单的线性探针: 输入隐藏状态,输出幻觉概率."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_state)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 Hallucination Probe 并保存为 .pt 文件")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset", type=str, default="tqa")
    parser.add_argument("--answers_dir", type=str,
                        default="save_for_eval/{dataset}_hal_det/answers",
                        help="存放 most_likely/batch_generations 的目录,支持 {dataset} 占位符")
    parser.add_argument("--answers_prefix", type=str,
                        default="most_likely_hal_det_{model}_{dataset}_answers_index_{idx}.npy",
                        help="答案文件命名模板,使用 {model},{dataset},{idx}")
    parser.add_argument("--bleurt_scores", type=str,
                        default="ml_{dataset}_bleurt_score.npy",
                        help="BLEURT 分数文件路径,支持 {dataset}")
    parser.add_argument("--bleurt_threshold", type=float, default=0.5,
                        help="BLEURT < threshold 视为 hallucination (label=1)")
    parser.add_argument("--layer_id", type=int, default=9)
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="用于训练探针的样本数量")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="若安装 bitsandbytes,可开启 4bit 量化以节省显存")
    parser.add_argument("--output_path", type=str, default="artifacts/probe_weights.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model_and_tokenizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ValueError("未安装 bitsandbytes, 无法 load_in_4bit")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )
    else:
        # 不用 device_map, 直接加载到 device
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


@torch.no_grad()
def extract_hidden_state(model, tokenizer, prompt: str, layer_id: int, device: torch.device) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states[layer_id]
    last_token = hidden_states[:, -1, :]  # [1, hidden_size]
    return last_token.squeeze(0).cpu().float()  # 转为 float32


def build_prompt(question: str, answer: str) -> str:
    return f"Answer the question concisely.\nQ: {question}\nA:{answer}"


def load_answers(base_dir: str, template: str, idx: int, model_name: str, dataset: str) -> List[str]:
    # 提取短模型名 (EleutherAI/gpt-neo-1.3B -> gpt-neo-1.3B)
    short_model = model_name.split('/')[-1] if '/' in model_name else model_name
    file_name = template.format(model=short_model, dataset=dataset, idx=idx)
    path = Path(base_dir) / file_name
    if not path.exists():
        return []
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        return list(arr.tolist())
    return arr.tolist()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    answers_dir = Path(args.answers_dir.format(dataset=args.dataset))
    bleurt_path = Path(args.bleurt_scores.format(dataset=args.dataset))
    if not bleurt_path.exists():
        raise FileNotFoundError(f"未找到 BLEURT 分数文件: {bleurt_path}")
    bleurt_scores = np.load(bleurt_path)
    
    # 先扫描有多少个答案文件,避免加载整个数据集
    print(f"扫描答案文件...")
    short_model = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    available_indices = []
    for idx in range(len(bleurt_scores)):
        file_name = args.answers_prefix.format(model=short_model, dataset=args.dataset, idx=idx)
        path = answers_dir / file_name
        if path.exists():
            available_indices.append(idx)
        if len(available_indices) >= args.max_samples:
            break
    
    if not available_indices:
        raise RuntimeError("未找到任何答案文件,请检查 answers_dir 和 answers_prefix 是否正确。")
    
    print(f"找到 {len(available_indices)} 个有效答案文件,开始加载模型...")
    model, tokenizer = load_model_and_tokenizer(args, device)

    # 仅需要 question 字段,尝试加载 TruthfulQA
    try:
        from datasets import load_dataset
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        questions = [sample["question"] for sample in dataset]
    except Exception as e:
        print(f"加载 TruthfulQA 数据集失败: {e}")
        print("使用占位符 question...")
        questions = [f"Question {i}" for i in range(len(bleurt_scores))]

    features: List[torch.Tensor] = []
    labels: List[int] = []

    for idx in available_indices:
        answers = load_answers(answers_dir, args.answers_prefix, idx, args.model_name, args.dataset)
        if not answers:
            continue
        answer = answers[0]
        question = questions[idx] if idx < len(questions) else f"Question {idx}"
        prompt = build_prompt(question, answer)
        hidden_vec = extract_hidden_state(model, tokenizer, prompt, args.layer_id, device)
        features.append(hidden_vec)
        bleurt_score = bleurt_scores[idx] if idx < len(bleurt_scores) else 0.0
        label = 1 if bleurt_score < args.bleurt_threshold else 0
        labels.append(label)
        
        if len(features) % 50 == 0:
            print(f"已处理 {len(features)} 个样本...")

    if not features:
        raise RuntimeError("未收集到任何样本,请检查 answers/bleurt 数据是否存在。")

    X = torch.stack(features)  # [N, hidden]
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    probe = HallucinationProbe(X.size(1)).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    num_samples = X.size(0)
    print(f"收集到 {num_samples} 条样本, 开始训练 probe ...")

    for epoch in range(args.epochs):
        perm = torch.randperm(num_samples)
        X_epoch = X[perm]
        y_epoch = y[perm]
        running_loss = 0.0

        for start in range(0, num_samples, args.batch_size):
            end = start + args.batch_size
            batch_x = X_epoch[start:end].to(device)
            batch_y = y_epoch[start:end].to(device)

            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / num_samples
        with torch.no_grad():
            logits = probe(X.to(device))
            preds = torch.sigmoid(logits).cpu()
            pred_labels = (preds >= 0.5).float()
            acc = (pred_labels.eq(y).float().mean()).item()
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.4f} - acc: {acc:.4f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), output_path)
    print(f"Probe 权重已保存至 {output_path}")


if __name__ == "__main__":
    main()


