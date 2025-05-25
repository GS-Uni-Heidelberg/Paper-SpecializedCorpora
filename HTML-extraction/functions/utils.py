import re
import os
import json
import shutil
import html2text


def markdown_to_text_re(markdown_string):
    # Handle bold and italic, ensuring escaped chars are preserved
    text = re.sub(r'\\\*', r'\\a', markdown_string)
    text = re.sub(r'\\_', r'\\b', text)

    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    # Remove links
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    text = re.sub(r'  +', ' ', text)

    # Remove single newlines
    lines = text.split('\n')
    text = '\n'.join([line.strip() for line in lines])
    text = re.sub(r'(?<!\n)\n(?!(\n)|( {0,}[*+-]))', ' ', text)

    new_lines = []
    for line in text.split('\n\n'):
        if line.strip().startswith('#'):
            raw_line = re.sub(r'^#+', '', line).strip()
            new_lines.extend(['', '', raw_line, ''])
        else:
            new_lines.append(line.strip())

    text = '\n\n'.join(new_lines)

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    return text


def html_to_text(html):

    parser = html2text.HTML2Text()
    parser.unicode_snob = True
    parser.ignore_emphasis = True
    parser.ignore_links = True
    parser.ignore_tables = True
    parser.mark_code = True

    md_text = parser.handle(html)
    raw_text = markdown_to_text_re(md_text)

    return raw_text


def rename_files_with_padded_index(directory):
    # Get list of files in the directory
    files = [
        file for file in os.listdir(directory)
        if not file.endswith('.py')
        and os.path.isfile(os.path.join(directory, file))
    ]

    # Calculate the total number of digits needed for the highest index
    num_files = len(files)
    num_digits = len(str(num_files))

    # Iterate over each file and rename it with a zero-padded index
    for index, filename in enumerate(files):
        if 'deduped' in filename:
            continue
        # Construct zero-padded index
        padded_index = str(index + 1).zfill(num_digits)

        # Construct new filename with padded index
        new_filename = f"{padded_index}.json"

        # Rename the file
        os.rename(
            os.path.join(directory, filename),
            os.path.join(directory, new_filename)
        )


def rename_files_with_padded_index_prefixed(directory, prefix):
    # Get list of files in the directory
    files = [
        file for file in os.listdir(directory)
        if not file.endswith('.py')
        and os.path.isfile(os.path.join(directory, file))
    ]

    # Calculate the total number of digits needed for the highest index
    num_files = len(files)
    num_digits = len(str(num_files))

    # Iterate over each file and rename it with a zero-padded index
    for index, filename in enumerate(files):
        if 'deduped' in filename:
            continue
        # Construct zero-padded index
        padded_index = str(index + 1).zfill(num_digits)

        # Construct new filename with padded index
        new_filename = f"{prefix}-{padded_index}.json"

        # Rename the file
        os.rename(
            os.path.join(directory, filename),
            os.path.join(directory, new_filename)
        )


def listdir_filetype(dirpath, filetype, absolute=False):
    files = [
        f for f in os.listdir(dirpath)
        if f.endswith(filetype)
    ]
    if absolute:
        files = [os.path.join(dirpath, f) for f in files]

    return files


def listdirf_filetype_gen(dirpath, filetype, absolute=False):
    for f in os.listdir(dirpath):
        if f.endswith(filetype):
            if absolute:
                yield os.path.join(dirpath, f)
            else:
                yield f


def jsondir_filepath_data_gen(dirpath, absolute=False):
    for f in listdirf_filetype_gen(dirpath, '.json', absolute=True):
        with open(f, 'r') as file:
            data = json.load(file)
        if not absolute:
            f = os.path.basename(f)
        yield f, data


def write_json(data, filepath, overwrite=False):
    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(f"File already exists: {filepath}")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def copy_jsoncorpus(
    src_filepath,
    dest_filepath,
    keys=None
):
    os.makedirs(dest_filepath, exist_ok=True)

    if keys:
        for f, data in jsondir_filepath_data_gen(src_filepath):
            data = {k: data[k] for k in keys}
            full_destpath = os.path.join(dest_filepath, f)
            if not os.path.exists(full_destpath):
                write_json(data, full_destpath)
            else:
                print(f"File already exists: {full_destpath}")
    else:
        for f in listdirf_filetype_gen(src_filepath, '.json', absolute=True):
            dst_path = os.path.join(dest_filepath, os.path.basename(f))
            if not os.path.exists(dst_path):
                shutil.copy(f, dest_filepath)
            else:
                print(f"File already exists: {dst_path}")
