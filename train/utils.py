from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from datasets import load_dataset
from functools import partial
import numpy as np

DATASET = 'tatsu-lab/alpaca'
MODEL_ID = 'gpt2'
END_KEY = '### End'
INSTRUCTION_KEY = '### Instruction:'
RESPONSE_KEY = '### Response:\n'
SEED = 42

PROMPT = """The following is an instruction that describes a task. Write a response that appropriately completes the request.

%s
{instruction}

%s""" % (INSTRUCTION_KEY, RESPONSE_KEY)

def load_model_and_tokenizer(location):
    model = AutoModelForCausalLM.from_pretrained(
        location,
        trust_remote_code = True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        location,
        trust_remote_code = True
    )
    return model, tokenizer

def create_response(
        instruction,
        model,
        tokenizer,
        do_sample = True,
        max_new_tokens = 256,
        top_p = 0.92,
        top_k = 0,
        **kwargs
):
    ids = tokenizer(PROMPT.format(instruction = instruction), return_tensors = 'pt').input_ids

    response_id = tokenizer.encode(RESPONSE_KEY)[0]
    end_id = tokenizer.encode(END_KEY)[0]

    tokens = model.generate(
        ids,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = end_id,
        do_sample = do_sample,
        max_new_tokens = max_new_tokens,
        top_p = top_p,
        top_k = top_k,
        **kwargs
    )[0].cpu()

    res_pos = np.where(tokens == response_id)[0]

    if len(res_pos) == 0:
        return None
    
    res_pos = res_pos[0]
    end_pos = np.where(tokens == end_id)[0]
    if len(end_pos) > 0:
        end_pos = end_pos[0]
    else:
        end_pos = None

    return tokenizer.decode(tokens[res_pos + 1 : end_pos]).strip()

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        res_tok_id = self.tokenizer.encode(RESPONSE_KEY)
        labels = batch['labels'].clone()

        for i in range(len(examples)):
            res_tok_id_start_idx = None
            for idx in np.where(batch['labels'][i] == res_tok_id[0])[0]:
                res_tok_id_start_idx = idx
                break
            labels[i, :res_tok_id_start_idx + 1] = -100

        batch['labels'] = labels

        return batch
    
def get_model_and_tokenizer(model_id = MODEL_ID, gradient_checkpointing = False):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code = True, use_cache = False if gradient_checkpointing else True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens' : [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY]})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch['text'],
        max_length = max_length,
        truncation = True
    )

def preprocess_dataset(tokenizer, max_length, dataset_name = DATASET, seed = SEED):
    dataset = load_dataset(dataset_name)['train']
    dataset = dataset.filter(lambda rec : not rec['text'].strip().endswith(RESPONSE_KEY.strip()))

    def _func(rec):
        rec['text'] += f'\n\n{END_KEY}'
        return rec

    dataset = dataset.map(_func)

    _preproc_func = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preproc_func,
        batched = True,
        remove_columns = ['instruction', 'input', 'output', 'text']
    )

    dataset = dataset.shuffle(seed = seed)
    return dataset

def train(
        local_output_dir,
        epochs,
        train_batch_size,
        eval_batch_size,
        lr,
        seed,
        gradient_checkpointing,
        test_size = 1000
):
    set_seed(seed)

    model, tokenizer = get_model_and_tokenizer(gradient_checkpointing = gradient_checkpointing)
    conf = model.config
    max_length = getattr(conf, 'n_positions', getattr(conf, 'seq_length', 1024))

    processed_dataset = preprocess_dataset(tokenizer, max_length)
    split_dataset = processed_dataset.train_test_split(test_size = test_size, seed = seed)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer = tokenizer,
        mlm = False,
        return_tensors = 'pt',
        pad_to_multiple_of = 8
    )

    training_args = TrainingArguments(
        output_dir = local_output_dir,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        learning_rate = lr,
        num_train_epochs = epochs,
        gradient_checkpointing = gradient_checkpointing,
        logging_dir = f'{local_output_dir}/runs',
        logging_strategy = 'steps',
        logging_steps = 10,
        evaluation_strategy = 'steps',
        eval_steps = 100,
        save_strategy = 'steps',
        save_steps = 200,
        save_total_limit = 1,
        load_best_model_at_end = True,
        report_to = 'tensorboard',
        disable_tqdm = True,
        remove_unused_columns = False,
    )

    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = split_dataset['train'],
        eval_dataset = split_dataset['test'],
        data_collator = data_collator
    )
    trainer.train()

    trainer.save_model(local_output_dir)
