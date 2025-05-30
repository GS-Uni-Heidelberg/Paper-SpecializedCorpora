import os
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from datetime import datetime
from .objects import SysPrompt, SysUserPrompt, NeighboursPrompt
from openai import OpenAI
from pydantic import BaseModel
from .metadata import RunMetadata
from dotenv import load_dotenv
from dataclasses import asdict
from collections import defaultdict


def default_kwargs(kwargs):
    """Return a dictionary with default values
    for the important kwargs that are not provided.
    """

    kwargs['temperature'] = kwargs.get("temperature", 0)
    kwargs['seed'] = kwargs.get("seed", 42)

    if not kwargs.get("model"):
        kwargs['model'] = "gpt-4o-mini-2024-07-18"
        print("No model provided. Using default model: gpt-4o-mini-2024-07-18")

    return kwargs


def get_openai_client(
    env_key: str = "API_KEY_SEMINAR",
    org_key: str = "ORG_SEMINAR"
):
    """Load environment variables and return OpenAI client."""
    load_dotenv()
    api_key = os.getenv(env_key)
    org = os.getenv(org_key)

    if not api_key:
        raise ValueError(f"Environment variable {env_key} not found")

    client = OpenAI(
        organization=org,
        api_key=api_key
    )
    return client


# +++ BATCH PROMPTING FUNCTIONS +++

def create_batch_entry(
    id,
    prompt: SysUserPrompt | NeighboursPrompt,
    content: str,
    response_format: BaseModel,
    neighbours: list[str] | None = None,
    **kwargs
):
    """Create a batch entry from a system prompt and a paragraph to annotate.
    To be used in a jsonl file.
    """

    kwargs = default_kwargs(kwargs)

    if neighbours is not None:
        userprompt = prompt.get_userprompt(content, neighbours)
    else:
        userprompt = prompt.get_userprompt(content)

    entry = {
        "custom_id": str(id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "messages": [
                {"role": "system", "content": prompt.get_sysprompt()},
                {"role": "user", "content": userprompt},
            ],
            "response_format": response_format.json_representation(),
        }
    }
    entry["body"].update(kwargs)

    return entry


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


def batch_request(
    instances: list,
    prompt: SysPrompt | SysUserPrompt,
    client: OpenAI,
    response_format: BaseModel,
    batch_filedir: str | Path,
    run_info: RunMetadata,
    neighbours: list[str] | None = None,
    **kwargs
):
    """Create a batch request for a system prompt and a dataset to annotate.
    """

    kwargs = default_kwargs(kwargs)

    batch = []
    data = []
    if 'custom_id' in instances[0]:
        raise ValueError(
            "Data already contains a custom_id field and cannot be used."
        )

    for index, instance in enumerate(instances):
        if neighbours is not None:
            nearest_neighbours = neighbours[index]
            entry = create_batch_entry(
                index, prompt, instance['text'],
                response_format,
                neighbours=nearest_neighbours,
                **kwargs
            )
        else:
            entry = create_batch_entry(
                index, prompt, instance['text'],
                response_format,
                **kwargs
            )
        batch.append(entry)
        data.append(create_data_entry(index, instance))

    temp_file = os.path.join(
        batch_filedir,
        'temp_files',
        f'temp_input-{datetime.now().strftime("%Y-%m-%d--%H%M%S")}.jsonl'
    )

    with open(temp_file, 'w') as f:
        for entry in batch:
            f.write(json.dumps(entry) + '\n')

    batch_file = client.files.create(
        file=open(temp_file, 'rb'),
        purpose='batch',
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    run_info = asdict(run_info)
    run_info['batch_id'] = batch_job.id
    data.insert(0, run_info)

    full_batchdata_filepath = os.path.join(
        batch_filedir,
        'data_files',
        f'batchdata_{datetime.now().strftime("%Y-%m-%d--%H%M%S")}.json'
    )
    with open(full_batchdata_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    return full_batchdata_filepath


def results_from_batch(batchdata_file, client):
    """Retrieve the results from a batch job.
    Creates a temporary file in the process.

    Args:
        batchdata_file (str): The path to the batch data file.
        client: The OpenAI client with permission
            to retrieve the batch results.
    """

    with open(batchdata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    batch_id = data[0]['batch_id']
    batch_metadata = data[0]
    print(
        f"Retrieving batch results for {batch_id}"
        f' with model: {batch_metadata["model"]}'
        f' and prompt name: {batch_metadata["prompt_name"]}'
        f' and output: {batch_metadata["output_format"]}'
    )
    data = data[1:]

    if 'label' in data[0]:
        for entry in data:
            entry['gold_label'] = entry.pop('label')

    status = client.batches.retrieve(batch_id).status == 'completed'

    if not status:
        print("Batch not yet completed.")
        return None

    batch_job = client.batches.retrieve(batch_id)
    result_file_id = batch_job.output_file_id
    result_file = client.files.content(result_file_id).content

    temp_output_file = os.path.join(
        os.path.dirname(os.path.dirname(batchdata_file)),
        'temp_files',
        f'temp_output-{datetime.now().strftime("%Y-%m-%d--%H%M")}.jsonl'
    )
    with open(temp_output_file, 'wb') as f:
        f.write(result_file)

    results = []
    with open(temp_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line.strip())
            results.append(json_object)

    for result in results:
        if result['response']['status_code'] != 200:
            data[int(result['custom_id'])]['output'] = 'FAIL'
        data[int(result['custom_id'])]['output'] = result['response']['body']

    for datum in data:
        if datum['output'] is None:
            datum['output'] = 'FAIL'

    fails = len([entry for entry in data if entry['output'] == 'FAIL'])
    print(f"Batch completed with {fails} unsuccessful completions.")

    return data


def save_batch(batchdata_file, client, final_file, key, **kwargs):
    """Converts the batch data to a final file.
    Creates temp files in the process.

    Args:
        batchdata_file (str): The path to the batch data file.
        client: The OpenAI client with permission
            to retrieve the batch results.
        final_file (str): The path to the final results file.
    """

    results = results_from_batch(batchdata_file, client)

    if results is None:
        print("Nothing was saved.")
        return None

    if os.path.exists(final_file):
        raise FileExistsError("Goal file already exists.")

    save_results(
        results, final_file, key, clean=True, **kwargs
    )


def loop_prompt(
    instances: list[dict],
    prompt: SysPrompt | SysUserPrompt,
    client: OpenAI,
    response_format: BaseModel,
    **kwargs
):
    kwargs = default_kwargs(kwargs)

    results = []
    for entry in tqdm(instances):
        messages = [
            {"role": "system", "content": prompt.get_sysprompt()},
            {"role": "user", "content": prompt.get_userprompt(entry['text'])}
        ]

        try:
            completion = client.beta.chat.completions.parse(
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
            results_dict = {
                'output': completion.to_dict(),
            }

        except Exception as e:
            print(f"Error with entry: {entry}")
            print(e)
            results_dict = {
                'output': 'ERROR',
            }

        results_dict = {**results_dict, **entry}
        results.append(results_dict)

    return results


def get_structured_output(completion):
    if completion == 'ERROR':
        return defaultdict(None)
    output = completion['output']['choices'][0]['message']['content']

    try:
        output = json.loads(output)
        return output
    except json.JSONDecodeError:
        print("Error decoding JSON")
        return defaultdict(None)


def get_prediction(completion, key):
    return get_structured_output(completion).get(key, False)


def gold_prediction_lists(results, key):
    gold_list = [entry.get('gold_label', '3_kein_thema') for entry in results]
    pred_list = [get_prediction(entry, key) for entry in results]
    return gold_list, pred_list


def get_confusion(index, gold, prediction):
    if gold[index] and prediction[index]:
        return "TP"
    if not gold[index] and not prediction[index]:
        return "TN"
    if not gold[index] and prediction[index]:
        return "FP"
    if gold[index] and not prediction[index]:
        return "FN"
    return "ERROR"


def remove_keys(original_dict, keys_to_remove):
    return {k: v for k, v in original_dict.items() if k not in keys_to_remove}


def keep_keys(original_dict, keys_to_keep):
    return {k: v for k, v in original_dict.items() if k in keys_to_keep}


def clean_results(results, key, keys_to_remove=[]):
    gold, pred = gold_prediction_lists(results, key)
    confusion = [get_confusion(i, gold, pred) for i in range(len(gold))]

    cleaned_results = []
    for i, result in enumerate(results):
        prediction = get_prediction(result, key)

        cot_elements = get_structured_output(result)
        cot_elements = remove_keys(cot_elements, [key])
        cot_elements = {
            f'cot_{i}': v for i, v in enumerate(cot_elements.values())
        }

        cleaned_result = remove_keys(result, ['output'] + keys_to_remove)
        cleaned_result['prediction'] = prediction
        cleaned_result['confusion'] = confusion[i]
        cleaned_result = cleaned_result | cot_elements

        cleaned_results.append(cleaned_result)

    return cleaned_results


def save_results(
    results, goal_path, key, clean=True, **kwargs
):

    if os.path.exists(goal_path):
        raise FileExistsError(f"File already exists at {goal_path}")

    if clean:
        results = clean_results(results, key, **kwargs)

    df = pd.DataFrame(results)
    df.to_csv(goal_path, index=False, sep='\t')

    return df
