import os
import json
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# +++++EXPLORING CRAWLING METADATA+++++

def document_count(directory, extension='.json'):
    count = 0
    for filename in os.listdir(directory):
        if (
            filename.endswith(extension)
            and os.path.isfile(os.path.join(directory, filename))
        ):
            count += 1
    return count


def vis_crawling_history(
    directory,
    date_retriever=lambda x: x['time_crawled']
):
    dates = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            try:
                dates.append(date_retriever(data))
            except KeyError:
                print(f"Error: No 'time_crawled' key in {filename}")
                print('Keys: ', data.keys())
    dates = [
        datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f").date()
        for dt_str in dates
    ]

    # Count occurrences of each date
    date_counts = Counter(dates)

    # Extract dates and counts for plotting
    sorted_dates = sorted(date_counts)
    counts = [date_counts[date] for date in sorted_dates]

    latest_date = max(dates)
    print('Cut-off date for scraping: ', latest_date)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_dates, counts, color='gray')
    plt.title('Number of Files per Date')
    plt.xlabel('Date')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_total_size(directory):
    total_size = 0
    # Iterate over all files in the directory
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            # Get the full path of the file
            filepath = os.path.join(dirpath, filename)
            # Get the size of the file in bytes
            file_size = os.path.getsize(filepath)
            # Add the file size to the total size
            total_size += file_size
    return total_size


def format_size(size):
    units = ['bytes', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


# +++++EXPLORING DOCUMENT METADATA+++++

def show_meta_elements(example_file):
    with open(example_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    soup = BeautifulSoup(data['html_content'], 'lxml')

    meta_data = soup.find_all('meta')

    for datum in meta_data:
        name = datum.get("name")
        if name:
            info = datum.get("content")
        else:
            name = datum.get("property")
            if name:
                info = datum.get("content")

        if name:
            print(f'Meta Element "{name}": {info}')

    print(
        '\nIf you find any of these useful, you can add',
        'their name (given in quotiation marks "")',
        'into the config.yaml section "metadata_keep".',
        '\nKeywords however will be retrieved automatically,',
        'no need to add that name to the config.'
    )
