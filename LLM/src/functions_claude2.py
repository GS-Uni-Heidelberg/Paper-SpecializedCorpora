"""This module contains utility functions for working with OpenAI's API.

TODO:
    + Improve eval file layouts?
    + Improve documentation.
    + Build in fail-safes for the loop_prompt function.
"""


import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from . import objects
from dataclasses import dataclass
from decimal import Decimal
import sklearn.metrics as eval
import json
from .objects import SysUserPrompt, NeighboursPrompt
from .metadata import RunMetadata
from pathlib import Path
from dataclasses import asdict


# +++ COST CALCULATION +++

@dataclass
class PricingSonnet:
    input_tokens: Decimal = Decimal('0.000003')
    output_tokens: Decimal = Decimal('0.000015')
    input_tokens_batch: Decimal = Decimal('0.0000015')
    output_tokens_batch: Decimal = Decimal('0.0000075')


@dataclass
class PricingOpus:
    input_tokens: Decimal = Decimal('0.000015')
    output_tokens: Decimal = Decimal('0.000075')
    input_tokens_batch: Decimal = Decimal('0.0000075')
    output_tokens_batch: Decimal = Decimal('0.0000375')


def get_pricing(model_str):
    if 'sonnet' in model_str:
        return PricingSonnet()
    if 'opus' in model_str:
        return PricingOpus()


def calc_cost_results(completions, model, batch=False):

    if isinstance(completions[0], dict):
        completions = [
            completion['output']
            for completion in completions
        ]

    costs = Decimal('0.0')
    for completion in completions:
        costs += calc_cost_instance(completion, model)

    if batch:
        costs = costs / 2

    return costs


def default_kwargs(kwargs):
    """Return a dictionary with default values
    for the important kwargs that are not provided.
    """

    kwargs['temperature'] = kwargs.get("temperature", 0)
    kwargs['max_tokens'] = kwargs.get("max_tokens", 2048)

    if not kwargs.get("model"):
        kwargs['model'] = "claude-3-5-sonnet-20240620"
        print(
            "No model provided. Using default model: "
            "claude-3-5-sonnet-20240620"
        )

    return kwargs


def remove_keys(original_dict, keys_to_remove):
    return {k: v for k, v in original_dict.items() if k not in keys_to_remove}


def keep_keys(original_dict, keys_to_keep):
    return {k: v for k, v in original_dict.items() if k in keys_to_keep}


def create_data_entry(
    id,
    orig_data,
    keys_to_keep: list[str] = [],
):
    """Create a data entry from a dictionary of data.
    To be used in a json file.
    """

    if len(keys_to_keep) > 0:
        entry = keep_keys(orig_data, keys_to_keep)
    else:
        entry = orig_data

    entry['custom_id'] = str(id)
    entry['output'] = None

    return entry


def calc_cost_instance(completion, model):
    output_tokens = completion.usage.output_tokens
    input_tokens = completion.usage.input_tokens
    pricing = get_pricing(model)
    cost = (
        pricing.input_tokens * Decimal(input_tokens)
        + pricing.output_tokens * Decimal(output_tokens)
    )
    return cost


# +++ BATCH PROMPTING +++

def create_batch_entry(
    id,
    prompt: SysUserPrompt | NeighboursPrompt,
    content: str,
    tool: str,
    tool_choice: dict,
    neighbours: list[str] = None,
    **kwargs
):

    kwargs = default_kwargs(kwargs)

    if neighbours is not None:
        userprompt = prompt.get_userprompt(content, neighbours)
    else:
        userprompt = prompt.get_userprompt(content)

    entry = {
        "custom_id": str(id),
        "params": {
            "system": prompt.get_sysprompt(),
            "tools": [tool],
            "tool_choice": tool_choice,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": userprompt
                        }
                    ]
                }
            ],
        }
    }

    entry['params'].update(kwargs)

    return entry


def read_expired_batch(
    batch_file: str | Path,
):
    # read jsonl as list of dicts
    with open(batch_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Find seen custom_ids
    seen_instances = []
    for entry in data:
        if entry['result']['type'] == 'succeeded':
            seen_instances.append(entry['custom_id'])

    return seen_instances


def batch_requests(
    instances: list,
    prompt: SysUserPrompt,
    client,
    tool,
    batch_filedir: str | Path,
    run_info: RunMetadata,
    tool_coice: dict = {
        "type": "tool", "name": "Tool"
    },
    neighbours: list[str] = None,
    drop_instances: list[str] = None,
    **kwargs
):

    kwargs = default_kwargs(kwargs)

    batch = []
    data = []
    for i, instance in enumerate(instances):
        paragraph = str(instance['text'])
        if neighbours is not None:
            nearest_neighbours = neighbours[i]
            batch.append(
                create_batch_entry(
                    i, prompt, paragraph, tool,
                    tool_choice=tool_coice,
                    neighbours=nearest_neighbours,
                    **kwargs,
                )
            )
        else:
            batch.append(
                create_batch_entry(
                    i, prompt, paragraph, tool,
                    tool_choice=tool_coice,
                    neighbours=None,
                    **kwargs,
                )
            )
        data.append(
            create_data_entry(i, instance)
        )

    # Remove instances that are already in the batch
    if drop_instances is not None:
        drop_instances = set(drop_instances)
        batch = [
            entry for entry in batch
            if entry['custom_id'] not in drop_instances
        ]
        data = [
            entry for entry in data
            if entry['custom_id'] not in drop_instances
        ]

    print(f"Batch size: {len(batch)}")

    message_batch = client.beta.messages.batches.create(requests=batch)

    run_info = asdict(run_info)
    run_info['batch_id'] = message_batch.id
    data.insert(0, run_info)

    full_batch_filepath = os.path.join(
        batch_filedir,
        f'batch_{datetime.now().strftime("%Y-%m-%d--%H%M%S")}.json'
    )

    with open(full_batch_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    return full_batch_filepath


def results_from_batch(batch_file, client):
    with open(batch_file, 'r') as f:
        data = json.load(f)

    batch_id = data[0]['batch_id']
    data = data[1:]

    retrieved = client.beta.messages.batches.retrieve(batch_id)

    fails = 0
    if retrieved.processing_status == 'ended':
        for result in client.beta.messages.batches.results(
            batch_id
        ):
            if result.result.type != 'succeeded':
                data[int(result.custom_id)]['output'] = 'FAIL'
                fails += 1
            data[int(result.custom_id)]['output'] = result.result.message

        print(f"Batch completed with {fails} unsuccessful completions.")

        return data

    print("Batch not yet completed.")

    return None


def results_from_file(
    batch_input_file: str | Path,
    batch_output_file: str | Path,
):
    # read jsonl as list of dicts
    with open(batch_output_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # read json input file as list of dicts
    with open(batch_input_file, 'r') as f:
        input_data = json.load(f)
        input_data = input_data[1:]

    # Align based on custom id
    data_dict = {entry['custom_id']: entry for entry in data}
    input_data_dict = {entry['custom_id']: entry for entry in input_data}
    for entry in input_data_dict:
        # combine the two dicts
        if entry in data_dict:
            input_data_dict[entry]['output'] = data_dict[entry]
        else:
            input_data_dict[entry]['output'] = 'FAIL'

    results = list(input_data_dict.values())

    return results


# +++ NORMAL PROMPTING +++

def loop_sysprompt(
    instances, sys_prompt, client, tool,
    model="claude-3-5-sonnet-20240620", temperature=0.3,
    tool_choice={"type": "tool", "name": "Tool"},
    max_tokens=512
):

    if isinstance(sys_prompt, objects.SysPrompt):
        sys_prompt = sys_prompt.sysprompt
    if not isinstance(sys_prompt, str):
        raise ValueError("sys_prompt must be a string or a SysPrompt object.")

    results = []
    for entry in tqdm(instances):
        label = entry['topic']
        paragraph = entry['text'],
        completion = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            temperature=temperature,
            tools=[tool],
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            system=sys_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": str(paragraph)
                        }
                    ]
                }
            ]
        )
        results_dict = {
            "paragraph": paragraph,
            "language": entry["language"],
            "genre": entry["genre"],
            "gold_label": bool(label),
            "output": completion
        }
        results.append(results_dict)
    return results


def loop_userprompt(
    instances, sysuserprompt, client, tool,
    model="claude-3-5-sonnet-20240620", temperature=0.3,
    tool_choice={"type": "tool", "name": "print_annotation"},
    max_tokens=512
):

    if not isinstance(sysuserprompt, objects.SysUserPrompt):
        raise ValueError("sysuserprompt must be a SysUserPrompt object.")

    results = []
    for entry in tqdm(instances):
        label = entry['topic']
        paragraph = (
            sysuserprompt.userprompt_begin
            + entry['text']
            + sysuserprompt.userprompt_end
        )
        completion = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            temperature=temperature,
            tools=[tool],
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            system=sysuserprompt.sysprompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": paragraph
                        }
                    ]
                }
            ]
        )
        results_dict = {
            "paragraph": paragraph,
            "language": entry["language"],
            "genre": entry["genre"],
            "gold_label": bool(label),
            "output": completion
        }
        results.append(results_dict)
    return results


# +++ EVAL FUNCTIONS +++

def gold_output_lists(results, bool_key="topic"):
    gold_list = [result["gold_label"] for result in results]
    output_list = [
        bool(result['output'].content[0].input[bool_key])
        for result in results
    ]

    return gold_list, output_list


def print_eval(results, bool_key="topic"):

    gold, output = gold_output_lists(results, bool_key)

    print(f"Accuracy: {eval.accuracy_score(
        gold, output)}")
    print(
        f"Macro F1: {eval.f1_score(
            gold, output, average='macro')}"
        )
    print(f"Micro F1: {eval.f1_score(
        gold, output, average='micro')}")
    print('+'*50)
    print(f"Precision 1class: {eval.precision_score(
        gold, output, pos_label=True)}")
    print(f"Recall 1class: {eval.recall_score(
        gold, output, pos_label=True)}")
    print(f"F1 1class: {eval.f1_score(
        gold, output, pos_label=True)}")
    print('+'*50)
    print(f"Precision 0class: {eval.precision_score(
        gold, output, average='binary', pos_label=False)}")
    print(f"Recall 0class: {eval.recall_score(
        gold, output, average='binary', pos_label=False)}")
    print(f"F1 0class: {eval.f1_score(
        gold, output, average='binary', pos_label=False)}")
    print('+'*50)
    print(
        "Confusion Matrix:",
        eval.confusion_matrix(gold, output)
    )


# +++ SAVING THE DATA +++

def clean_results(
    results,
    cot=False, bool_key="topic", explanation_key="explanation"
):

    gold, output = gold_output_lists(results, bool_key)

    def get_confusion(index):
        if gold[index] and output[index]:
            return "TP"
        if not gold[index] and not output[index]:
            return "TN"
        if not gold[index] and output[index]:
            return "FP"
        if gold[index] and not output[index]:
            return "FN"
        return "ERROR"

    cleaned_results = [
        {
            "paragraph": result["paragraph"],
            "language": result["language"],
            "genre": result["genre"],
            "gold_label": result["gold_label"],
            "prediction": result["output"].content[0].input.get(
                bool_key
            ),
            "explanation": result["output"].content[0].input.get(
                explanation_key
            ) if cot else None,
            "confusion": get_confusion(i),
        }
        for i, result in enumerate(results)
    ]
    return pd.DataFrame(cleaned_results)


def results_to_csv(
    results, csv_file,
    **kwargs
):
    cleaned_results_df = clean_results(results, **kwargs)

    cleaned_results_df.to_csv(csv_file)

    return cleaned_results_df
