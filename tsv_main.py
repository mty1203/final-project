import os
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
from train_utils import get_last_non_padded_token_rep, compute_ot_loss_cos, update_centroids_ema, update_centroids_ema_hard, get_ex_data, collate_fn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llm_layers import add_tsv_layers
from sklearn.metrics import roc_auc_score

# Check if bitsandbytes is installed (for optional 4-bit quantization)
try:
    import bitsandbytes as bnb  # noqa: F401
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
try:
    # New PyTorch API (torch >= 1.10)
    from torch.amp import autocast
    USE_NEW_AMP = True
except ImportError:
    # Legacy PyTorch API
    from torch.cuda.amp import autocast
    USE_NEW_AMP = False

from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from sinkhorn_knopp import SinkhornKnopp_imb
import logging


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_tsv_vectors(tsv_params: nn.ParameterList, args, path: str):
    if not path:
        return
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    payload = {
        "tsv_vectors": [param.detach().cpu() for param in tsv_params],
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "component": args.component,
        "str_layer": args.str_layer,
        "lam": args.lam,
    }
    torch.save(payload, path)
    logging.info(f"TSV vectors saved to {path}")


def train_model(model, optimizer, device, prompts, labels, args):
    
    layer_number = -1
    dir_name = f"TSV_{args.model_name}_{args.dataset_name}/exemplar_num_{args.num_exemplars}_num_selected_data_{args.num_selected_data}/{args.component}/{args.str_layer}/{args.lam}"
    log_dir = f"./{dir_name}/"
    log_file = os.path.join(log_dir, f"log.txt")
    os.makedirs(dir_name,exist_ok=True)
    
    logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",)
    
    logging.info("Starting training")
    logging.info(f"Training parameters: few_shot_size={args.num_exemplars}, num_selected_data={args.num_selected_data}, component={args.component}, str_layer={args.str_layer}")
    
    test_prompts, train_prompts, exemplar_prompts = prompts[0], prompts[1], prompts[2]
    test_labels, train_labels, exemplar_labels = labels[0], labels[1], labels[2]
    batch_size = args.batch_size
    num_samples = len(train_prompts)
    
    losses = []
    best_test_auroc = -1

    scaler = GradScaler()

    num_exemplars = args.num_exemplars

    # Initialize Sinkhorn algorithm
    args.num_iters_sk = 3
    args.epsilon_sk = 0.05
    
    ex_hallu = (num_exemplars-exemplar_labels[:num_exemplars].sum())/num_exemplars
    ex_true = (exemplar_labels[:num_exemplars].sum())/num_exemplars
    cls_dist = torch.tensor([ex_hallu,ex_true]).float().cuda()
    cls_dist = cls_dist.view(-1, 1) 
    sinkhorn = SinkhornKnopp_imb(args, cls_dist)
   
    # Initialize Centroids
    centroids = torch.randn((2, model.config.hidden_size)).half().cuda() 
    centroids = F.normalize(centroids, p=2, dim=1)   
    
    exemplar_prompts_, exemplar_labels_ = exemplar_prompts, exemplar_labels    
    exemplar_prompts, exemplar_labels = collate_fn(exemplar_prompts, exemplar_labels) 

    num_epochs = args.init_num_epochs
        
    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        all_labels = []
        num_samples = num_exemplars 

        # Process data in batches
        for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{num_epochs} Batches", leave=False):

            batch_prompts = exemplar_prompts[batch_start: batch_start + batch_size]
            batch_labels = exemplar_labels[batch_start: batch_start + batch_size]
            
            # Create attention masks (1 for real tokens, 0 for padding)
            attention_mask = (batch_prompts != 0).half() 
            
            batch_prompts = batch_prompts.to(device)
            batch_labels = batch_labels.to(device)
            attention_mask = attention_mask.to(batch_prompts.device)
            
            # Forward pass
            autocast_context = autocast('cuda', dtype=torch.float16) if USE_NEW_AMP else autocast(dtype=torch.float16)
            with autocast_context:
                # Use hidden_states list for compatibility with CausalLMOutput (e.g., GPT-Neo)
                output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                # Get hidden state from specified layer (layer_number=-1 means the last layer)
                last_layer_hidden_state = output.hidden_states[layer_number]  # [batch_size, seq_len, hidden_size]
                
                # Use attention mask to ignore padding tokens, and get the last non-padded token's representation
                last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())  
                
                batch_labels_oh = torch.nn.functional.one_hot(batch_labels, num_classes=-1)
                
                ot_loss, similarities = compute_ot_loss_cos(last_token_rep, centroids, batch_labels_oh, batch_size, args)
                
                loss = ot_loss 
                
                total += batch_labels.size(0)
                
                with torch.no_grad():
                     centroids = update_centroids_ema_hard(centroids, last_token_rep, batch_labels_oh, args)
                    
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            running_loss += loss.item() * batch_labels.size(0) 
          
        # Epoch summary
        epoch_loss = running_loss / total  

        if (epoch+1) % 1 == 0:
            
            test_labels_ = test_labels
            test_predictions, test_labels_combined = test_model(model, centroids, test_prompts, test_labels_, device, batch_size, layer_number)
            
            test_auroc = roc_auc_score(test_labels_combined.cpu().numpy(), test_predictions.cpu().numpy())  
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            losses.append(epoch_loss)
           
            # AUROC Calculation using sklearn
            test_predictions = test_predictions.cpu().numpy()
            test_labels_combined = test_labels_combined.cpu().numpy()

            if test_auroc > best_test_auroc:
                best_test_auroc = test_auroc
                best_test_epoch = epoch
                print(f"Best test AUROC: {best_test_auroc:.4f}, at epoch: {best_test_epoch }")
                logging.info(f"Best test AUROC: {best_test_auroc:.4f}, at epoch: {best_test_epoch }")

            logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, ")
            
            logging.info(f"Test AUROC: {test_auroc:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}],Test AUROC: {test_auroc:.4f}")           
    
    logging.info(f"SS Learning Starts")
    
    with torch.no_grad():
   
        selected_indices, selected_labels_soft = get_ex_data(model, train_prompts, train_labels, batch_size, centroids, sinkhorn, args.num_selected_data, cls_dist, args)
        
        num_samples = len(selected_indices) + args.num_exemplars
        
    num_epochs = args.aug_num_epochs
    
    exemplar_label = torch.tensor(exemplar_labels).cuda()     

    selected_prompts = [train_prompts[i] for i in selected_indices] 
    selected_labels = selected_labels_soft
    
    augmented_prompts = selected_prompts + exemplar_prompts_
    exemplar_labels = torch.nn.functional.one_hot(exemplar_label.to(torch.int64), num_classes=2)
    augmented_labels = torch.concat((selected_labels, torch.tensor(exemplar_labels).clone().cuda()))
    
    augmented_prompts_train = augmented_prompts
    augmented_labels_label = augmented_labels
    
    num_samples = len(augmented_prompts_train)
    
    # Adaptive old/new AMP API
    autocast_context = autocast('cuda', dtype=torch.float16) if USE_NEW_AMP else autocast(dtype=torch.float16)
    with autocast_context:
        for epoch in range(num_epochs):
            running_loss = 0.0
            total = 0
            all_labels = []
            
            for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{num_epochs} Batches", leave=False):
               
                batch_prompts = augmented_prompts_train[batch_start: batch_start + batch_size]
                batch_labels = augmented_labels_label[batch_start: batch_start + batch_size]

                batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)

                attention_mask = (batch_prompts != 0).half()  # Shape: [batch_size, max_seq_len]

                batch_prompts = batch_prompts.to(device)
                batch_labels = batch_labels.to(device)
                attention_mask = attention_mask.to(batch_prompts.device)

                # Use hidden_states list for specified layer output (compatible with CausalLMOutput)
                output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                last_layer_hidden_state = output.hidden_states[layer_number]  # Shape: [batch_size, max_seq_len, hidden_size]

                # Use attention mask to ignore padding tokens, and get the last non-padded token's representation
                last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())  # Shape: [batch_size, hidden_size]

                
                ot_loss, similarities = compute_ot_loss_cos(last_token_rep, centroids, batch_labels, batch_size, args)
                
                loss = ot_loss 

                with torch.no_grad():
                    
                   centroids = update_centroids_ema(centroids, last_token_rep, batch_labels.half(), args)
                   all_labels.append(batch_labels.cpu()) 
                   total += batch_labels.size(0)
                   
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Accumulate the loss
                running_loss += loss.item() * batch_labels.size(0)  

            epoch_loss = running_loss / total  # Normalize loss by total samples
            
            
            with torch.no_grad():
                all_labels = torch.cat(all_labels).numpy()
                test_labels_ = test_labels
             
                if epoch % 1 ==0:
                   test_predictions, test_labels_combined = test_model(model, centroids, test_prompts, test_labels_, device, batch_size, layer_number)
                   test_auroc = roc_auc_score(test_labels_combined, test_predictions)
                   
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            losses.append(epoch_loss)
        
            if test_auroc > best_test_auroc:
                best_test_auroc = test_auroc
                best_test_epoch = epoch + args.init_num_epochs
                #best_epoch = epoch + 1  # Storing epoch in 1-based index
                print(f"Best test AUROC: {best_test_auroc:.4f}, at epoch: {best_test_epoch}")
                logging.info(f"Best test AUROC: {best_test_auroc:.4f}, at epoch: {best_test_epoch}")
                  
            logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, ")
           
            logging.info(f"Best test AUROC: {best_test_auroc:.4f}, at epoch: {best_test_epoch}")
            
        return best_test_auroc


def test_model(model, centroids, test_prompts, test_labels, device, batch_size, layer_number):
    model.eval() 
    val_predictions = []
    val_labels_combined = []
 
    all_last_token_reps = []
    all_labels = []

    num_val_samples = len(test_prompts)
    
    with torch.no_grad():
        autocast_context = autocast('cuda', dtype=torch.float16) if USE_NEW_AMP else autocast(dtype=torch.float16)
        with autocast_context:
            for batch_start in range(0, num_val_samples, batch_size):
                batch_prompts = test_prompts[batch_start:batch_start + batch_size]
                batch_labels = test_labels[batch_start:batch_start + batch_size]
                batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)
                
                attention_mask = (batch_prompts != 0).half().to(device)
                batch_prompts = batch_prompts.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass: use hidden_states list to get specified layer output
                output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                last_layer_hidden_state = output.hidden_states[layer_number]
                last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())   
                
                all_last_token_reps.append(F.normalize(last_token_rep,p=2,dim=-1).detach().cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
                
                last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
                centroids = F.normalize(centroids, p=2, dim=-1)
                
                similarities = torch.matmul(last_token_rep, centroids.T)  # Shape: [batch, 2]

                similarity_scores  = torch.softmax(similarities/ 0.1, dim=-1)
                similarity_scores  = similarity_scores[:,1] 
                val_predictions.append(similarity_scores.cpu())
                val_labels_combined.append(batch_labels.cpu())
      

    val_predictions = torch.cat(val_predictions)
    val_labels_combined = torch.cat(val_labels_combined)
    
    return val_predictions, val_labels_combined


HF_NAMES = {
    # Original 8B LLaMA3.1 (requires large VRAM)
    'llama3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
    'qwen2.5-7B': 'Qwen/Qwen2.5-7B',
    # Smaller open-source models: GPT-Neo series (2.7B / 1.3B), suitable for 16GB VRAM
    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
    'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
}

def main(): 

    parser = argparse.ArgumentParser()
    # Default to smaller GPT-Neo-1.3B for training on 16GB VRAM
    parser.add_argument('--model_name', type=str, default='gpt-neo-1.3B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--num_gene', type=int, default=1)
    parser.add_argument('--gene', type=int, default=0) 
    parser.add_argument('--generate_gt', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='tqa')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--wild_ratio', type=float, default=0.75)
    parser.add_argument('--thres_gt', type=float, default=0.5)
    parser.add_argument('--most_likely', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cos_temp", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--str_layer", type=int, default=9)
    parser.add_argument("--component", type=str, default='res')
    parser.add_argument("--lam", type=float, default=5)
    parser.add_argument("--init_num_epochs", type=int, default=20)
    parser.add_argument("--aug_num_epochs", type=int, default=20)
    parser.add_argument("--num_exemplars", type=int, default=16)
    parser.add_argument("--num_selected_data", type=int, default=32)
    parser.add_argument("--cls_dist", type=str, default='proxy')
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--num_iters_sk", type=int, default=3)
    parser.add_argument("--epsilon_sk", type=float, default=0.05)
    parser.add_argument("--save_tsv_path", type=str, default=None, help="Optional: save TSV vectors to this path (.pt) after training")
    
    args = parser.parse_args()
        
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    # Set max memory for large models; excess weights are offloaded to CPU
    # Note: the GPU key must be integer 0 (not string "0")
    max_memory = {
        0: "10GiB",       # GPU0 uses at most 10GiB
        "cpu": "32GiB",   # Reduce if your system RAM is smaller
    }

    # If bitsandbytes is available, configure 4-bit quantization with fp16 compute to reduce VRAM
    quantization_config = None
    if HAS_BNB:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if args.dataset_name == "tqa":
        dataset = load_dataset("truthful_qa", 'generation')['validation']
    
    elif args.dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
    
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
        
        
    elif args.dataset_name == 'sciq':
        dataset = load_dataset("allenai/sciq", split="validation")
        
    elif args.dataset_name == 'nq_open':
        dataset = load_dataset("google-research-datasets/nq_open", split="validation") 
        
            
    else:
        raise ValueError("Invalid dataset name")
    

    if args.gene:

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token = '')
        # If bitsandbytes is available, use 4-bit quantization; otherwise fallback to FP16 + max_memory
        if quantization_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
                device_map="auto",
                quantization_config=quantization_config,
                max_memory=max_memory,
                token='',
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                token='',
            )
        device = torch.device("cuda")
        all_decoded_answers = []
        begin_index = 0
        end_index = len(dataset)
        
        if not os.path.exists(f'./save_for_eval/{args.dataset_name}_hal_det/'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}_hal_det/')

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}_hal_det/answers'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}_hal_det/answers')

        period_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n']]
        period_token_id += [tokenizer.eos_token_id]
        
        for i in range(begin_index, end_index):
            answers = [None] * args.num_gene
            answers_ = [None] * args.num_gene
            
            question = dataset[i]['question']
            prompt = tokenizer(f"Answer the question concisely. Q: {question}" + " A:", return_tensors='pt').input_ids.cuda()
            
            for gen_iter in range(args.num_gene):
                if args.most_likely:
                    generated = model.generate(prompt,
                                                num_beams=5,
                                                num_return_sequences=1,
                                                do_sample=False,
                                                max_new_tokens=64,
                                               )
                else:
                    generated = model.generate(prompt,
                                                do_sample=True,
                                                num_return_sequences=1,
                                                num_beams=1,
                                                max_new_tokens=64,
                                                temperature=0.5,
                                                top_p=1.0)
                
                
                decoded = tokenizer.decode(generated[0, prompt.shape[-1]:],
                                           skip_special_tokens=True)
                # answers[gen_iter] = decoded
     
                # Cleaning
                if '\nAnswer the question concisely.' in decoded:
                    print('#####error')
                    print(decoded.split('\nAnswer the question concisely.')[1])
                    print('#####error')
                    decoded = decoded.split('\nAnswer the question concisely.')[0]
                
                if 'Answer the question concisely' in decoded:
                    print('#####error')
                    print(decoded.split('Answer the question concisely')[1])
                    print('#####error')
                    decoded = decoded.split('Answer the question concisely')[0]
                    
                if 'The answer to the question' in decoded:
                    print('#####error')
                    print(decoded.split('The answer to the question')[1])
                    print('#####error')
                    decoded = decoded.split('The answer to the question')[0]
                
                if 'How to Write a Concise Statement' in decoded:
                    print('#####error')
                    print(decoded.split('How to Write a Concise Statement')[1])
                    print('#####error')
                    decoded = decoded.split('How to Write a Concise Statement')[0]         
                    
                if 'Q:' in decoded:
                    print('#####error')
                    print(decoded.split('Q:')[1])
                    print('#####error')
                    decoded = decoded.split('Q:')[0]     

                if '\nYou are an AI assistant' in decoded:    
                        print('#####error')
                        print(decoded.split('\nYou are an AI assistant')[1])
                        print('#####error')
                        decoded = decoded.split('\nYou are an AI assistant')[0]  
                        
                if 'You are an AI assistant' in decoded:    
                    print('#####error')
                    print(decoded.split('You are an AI assistant')[1])
                    print('#####error')
                    decoded = decoded.split('You are an AI assistant')[0]  
                    
                if 'A:' in decoded:
                    print('#####error')
                    print(decoded.split('A:')[1])
                    print('#####error')
                    decoded = decoded.split('A:')[0]  
                
                if 'B:' in decoded:
                    print('#####error')
                    print(decoded.split('B:')[1])
                    print('#####error')
                    decoded = decoded.split('B:')[0] 
                    
                if 'C:' in decoded:
                    print('#####error')
                    print(decoded.split('C:')[1])
                    print('#####error')
                    decoded = decoded.split('C:')[0] 
                    
                if 'D:' in decoded:
                    print('#####error')
                    print(decoded.split('D:')[1])
                    print('#####error')
                    decoded = decoded.split('D:')[0] 
                
                print(f'Cleaned Answer: {decoded}')
                answers[gen_iter] = decoded        
              
            
            
            print('sample: ', i)
            if args.most_likely:
                info = 'most_likely_'
            else:
                info = 'batch_generations_'
                
            print("Saving answers")
            print(decoded)
            
            np.save(f'./save_for_eval/{args.dataset_name}_hal_det/answers/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy',
                        answers)
            
    elif args.generate_gt:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

        model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').cuda()
        tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
        model.eval()
        
        gts = np.zeros(0)
        length = len(dataset)
            
        for i in range(length):
            
            if args.dataset_name == 'tqa':
                best_answer = dataset[i]['best_answer']
                correct_answer = dataset[i]['correct_answers']
                all_answers = [best_answer] + correct_answer
                question = dataset[i]['question']

            elif args.dataset_name == 'triviaqa':
                all_answers = dataset[i]['answer']['aliases']
                
            if args.most_likely:
                # answers = np.load(
                #     f'./save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
                answers = np.load(
                    f'./save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')

            else:
                answers = np.load(
                    f'./save_for_eval/{args.dataset_name}_hal_det/answers/batch_generations_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
                    
            # get the gt.
            predictions = answers
            all_results = np.zeros((len(all_answers), len(predictions)))
            with torch.no_grad():
                for anw in range(len(all_answers)):
                    inputs = tokenizer(predictions.tolist(), [all_answers[anw]] * len(predictions),
                                        padding='longest', return_tensors='pt')
                    for key in list(inputs.keys()):
                        inputs[key] = inputs[key].cuda()
                    res = np.asarray(model(**inputs).logits.flatten().tolist())
                    all_results[anw] = res
            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
            if i % 10 == 0:
                print("samples passed: ", i)
        
        if args.most_likely:
            # np.save(f'./ml_{args.dataset_name}_bleurt_score.npy', gts)
            np.save(f'./ml_{args.dataset_name}_bleurt_score.npy', gts)
            
        else:
            np.save(f'./bg_{args.dataset_name}_bleurt_score.npy', gts)
    

                
    else:
        
        device = torch.device("cuda")
        # Training phase: prefer 4-bit quantization + max_memory; otherwise FP16 + max_memory
        if quantization_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
                device_map="auto",
                quantization_config=quantization_config,
                max_memory=max_memory,
                token='',
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                token='',
            )

        # Disable KV cache during training to save VRAM
        if hasattr(model, "config"):
            model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token = '')
        
        prompts = []
        qa_pairs = []
        categories = []
        
        length = len(dataset)
    
        
        for i in tqdm(range(length)):
           
            question = dataset[i]['question']
            if args.dataset_name == 'tqa':
                categories.append(dataset[i]['category'])
            
            answers = np.load(
                f'./save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
            
     
            for anw in answers:

                prompt = tokenizer(
                    f"Answer the question concisely. Q: {question}" + " A:" + anw,
                    return_tensors='pt').input_ids.cuda()
                   
                prompts.append(prompt)    
                qa_pairs.append({'Question': question, 'Answer': anw})
        
        gts = np.load(f'./ml_{args.dataset_name}_bleurt_score.npy') 

        
        length = len(dataset)
        
        if args.dataset_name == 'tqa' or args.dataset_name == 'triviaqa':
            args.thres_gt = 0.5
        
        else:
            args.thres_gt = 0.2
               
        gt_label = np.asarray(gts> args.thres_gt, dtype=np.int32)
        
        # index = np.random.permutation(length)
        
        # exemplar_index = index[:args.num_exemplars]
        
        # wild_q_indices = index[:int(args.wild_ratio * length)]
        
        index = np.load(f'data_indices/data_index_{args.dataset_name}.npy')
        
        exemplar_index = np.load(f'data_indices/exemplar_idx_{args.dataset_name}.npy')
        
        wild_q_indices = index[:int(args.wild_ratio * length)]
    
        wild_q_indices1 = wild_q_indices[:len(wild_q_indices) - 100]
        
        args.num_exemplars = len(exemplar_index)
        
        gt_label_test = []
        gt_label_wild = []
        gt_label_exemplar = []
        
        test_prompts = []
        train_prompts = []
        exemplar_prompts = []
        
        
        for i in range(length):
            if i not in wild_q_indices:
                gt_label_test.extend(gt_label[i: i+1])
                test_prompts.extend(prompts[i:i+1])
                
            elif i in exemplar_index:
                gt_label_exemplar.extend(gt_label[i: i+1])
                exemplar_prompts.extend(prompts[i:i+1])
                
            elif i in wild_q_indices1:
                gt_label_wild.extend(gt_label[i: i+1])
                train_prompts.extend(prompts[i:i+1])
            
        gt_label_test = np.asarray(gt_label_test)  
        gt_label_exemplar = np.asarray(gt_label_exemplar)
        gt_label_wild = np.asarray(gt_label_wild)        

        labels = [ gt_label_test,  gt_label_wild, gt_label_exemplar]
        prompts = [ test_prompts,  train_prompts, exemplar_prompts]
        
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        
        for param in model.parameters():
                param.requires_grad = False
        
        tsv = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size), requires_grad=True) for _ in range(num_layers)])
        
        tsv.to(device)
        
        add_tsv_layers(model, tsv, [args.lam], args)
    
        optimizer = torch.optim.AdamW(list(tsv.parameters()), lr=args.lr)

        train_model(model, optimizer, device, prompts, labels, args=args)

        if args.save_tsv_path:
            save_tsv_vectors(tsv, args, args.save_tsv_path)

if __name__ == '__main__':
    seed_everything(42)
    main()
