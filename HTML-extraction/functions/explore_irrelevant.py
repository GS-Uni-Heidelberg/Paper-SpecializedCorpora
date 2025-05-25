import json
import os
import re
from tqdm import tqdm


def get_subdomains(urls, print_=False):
    """Given an array of urls, return a set of subdomains.

    Can also print the subdomains.
    """

    subdomains = {}
    for url in urls:
        subdomain_list = url.split('/')
        for subdomain in subdomain_list[:-1]:
            subdomains[subdomain] = subdomains.get(
                subdomain, 0
            ) + 1

    if print_:
        for subdomain, count in subdomains.items():
            print(subdomain, count)

    return subdomains


def get_url_ends(urls):
    url_ends = set()
    for url in urls:
        if url.endswith('/'):
            url_ends.add(url.split('/')[-2])
        else:
            url_ends.add(url.split('/')[-1])
    return url_ends


def bytepairs(urls, iterations):
    inventory = set()
    translations = {}
    iteration = 0

    for end in urls:
        for char in end:
            if char not in inventory:
                inventory.add(char)
                translations[char] = iteration
                iteration += 1

    url_ends = [[translations[char] for char in end] for end in urls]

    for n in range(iteration, iteration+iterations):
        pairs = {}
        for end in url_ends:
            for i in range(len(end) - 1):
                pairs[(end[i], end[i + 1])] = pairs.get(
                    (end[i], end[i + 1]), 0
                ) + 1
        highest_pair = max(pairs, key=pairs.get)
        inventory.add(highest_pair)
        translations[highest_pair] = n

        new_url_ends = []
        for end in url_ends:
            new_end = []
            skip = False
            for i, char in enumerate(end):
                if skip:
                    skip = False
                    continue
                if (
                    char == highest_pair[0]
                    and i < len(end) - 1 and end[i + 1] == highest_pair[1]
                ):
                    new_end.append(n)
                    skip = True
                else:
                    new_end.append(char)
            new_url_ends.append(new_end)
        url_ends = new_url_ends

    return translations


def get_common_bytepairs(
    urls, iterations=100,
    print_=False, min_print_len=2, full_url=False
):
    if not full_url:
        url_ends = get_url_ends(urls)
    else:
        url_ends = urls

    translations = bytepairs(url_ends, iterations)

    backtranslations = backtranslate(translations)
    if print_:
        for transl in backtranslations:
            if len(transl) >= min_print_len:
                print(transl)

    return backtranslations


def flatten_tuples(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, tuple):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list


def backtranslate(translations):
    inverted_transl = {v: k for k, v in translations.items()}
    backtranslations = set()
    for key in translations:
        if isinstance(key, str):
            backtranslations.add(key)
        else:
            pair = [key[0], key[1]]
            while True:
                if all(isinstance(element, str) for element in pair):
                    break
                for i, element in enumerate(pair):
                    if isinstance(element, int):
                        pair[i] = inverted_transl[element]
                pair = flatten_tuples(pair)
            backtranslations.add(''.join(pair))
    return backtranslations


def list_key(json_dir, key):
    """Lists all values of a key in a directory of json files."""
    urls = set()
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    urls.add(data[key])
    return urls


def matching_set(set_, pattern):
    """Gets the set of items in an array containing a regex pattern."""
    matches = set()
    for item in set_:
        if re.search(pattern, item):
            matches.add(item)
    return matches


def load_patterns(filepath):
    """Load a list of patterns from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        patterns = f.readlines()
    patterns = [pattern.strip('\n') for pattern in patterns]
    full_pattern = '|'.join(patterns)
    return full_pattern


def is_relevant(url, irrel_pattern):
    """Check if a url is relevant based on a pattern."""
    return not re.search(irrel_pattern, url)


def apply_function_dir(
    directory, function, iterations=None,
):
    """Apply a function to all files in a directory."""
    matching_files = []
    i = 0

    for root, _, files in os.walk(directory):
        for file in tqdm(
            files,
            desc=f'Processing {directory}', total=len(files)
        ):
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                match = function(data)
                if match:
                    matching_files.append(file)
            i += 1
            if iterations and i >= iterations:
                break
    return matching_files


def print_match_results(
    matches, filecount,
    list_matches=False, make_url=False
):
    if list_matches:
        for match in matches:
            if make_url:
                match = match.replace('_', '/')
                match = match.replace('.json', '')
            print(match)

    print(f"Number of matching files: {len(matches)}")
    print(f'This represents {len(matches)/filecount:.2%} of the files.')
