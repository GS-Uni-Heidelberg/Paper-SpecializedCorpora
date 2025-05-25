"""
PLACE FUNCTIONS IN THIS FILE THAT CHECK FOR CERTAIN CRITERIA IN THE DATA.

The functions take a dictionary as input and return a boolean value:
    True: The criteria is met.
    False: The criteria is not met -- the article should be discarded.
"""


import re
from bs4 import BeautifulSoup as BS
import htmldate


LAST_TEN_YEARS = r'201[4-9]|202[0-4]'
UNTIL_APRIL_2024 = r'Januar|Februar|März|April'


def infakt_is_february(data):

    html = data['html_content']
    # Check if the date is in first week or second of February 2020-2025
    years = {'2024', '2023', '2022', '2021', '2020', '2025'}
    months = {'Februar'}
    days = {
        '01', '02', '03', '04', '05', '06', '07',
        '08', '09', '10', '11', '12', '13', '14',
        '15', '16', '17', '18', '19', '20', '21',
        '22', '23', '24', '25', '26', '27', '28'
    }

    try:
        soup = BS(html, 'lxml')
        spans = soup.find_all('span')

        date = '  '
        for span in spans:
            if 'last_update' in span.get('id', ['']):
                date = span.get_text()
                break

        date = re.sub(r'\.', '', date)
        day, month, year = date.split(' ')

        return year in years and month in months and day in days
    except Exception:
        try:
            date = htmldate.find_date(html)
            year, month, day = date.split('-')

            return year in years and month in months and day in days
        except Exception:
            return False


def is_february(data):
    html = data['html_content']

    try:
        date = htmldate.find_date(html)
        year, month, day = date.split('-')

        # Check if the date is in first week or second of February 2020-2024
        years = {'2024', '2023', '2022', '2021', '2020', '2025'}
        months = {'02'}
        days = {
            '01', '02', '03', '04', '05', '06', '07',
            '08', '09', '10', '11', '12', '13', '14',
            '15', '16', '17', '18', '19', '20', '21',
            '22', '23', '24', '25', '26', '27', '28'
        }
        return year in years and month in months and day in days
    except Exception:
        return False


def zeit_content_tier(data):
    html = data['html_content']

    # find content tier meta tag
    try:
        soup = BS(html, 'lxml')
        meta = soup.find('property', {'name': 'article:content_tier'})
        if meta['content'] == 'locked':
            return False
        return True
    except Exception:
        return True


def zeit_dpa(data):
    html = data['html_content']

    if re.search(r'© dpa-infocom|Sie wurde automatisch von der Deutschen Presse-Agentur \(dpa\) übernommen\.', html):
        return False
    return True
