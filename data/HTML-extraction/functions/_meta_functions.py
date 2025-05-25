"""
PLACE FUNCTIONS HERE THAT EXTRACT METADATA (not meta-tagged elements,
but anything you find important) FROM THE HTML CONTENT.
ALSO, PLACE FUNCTIONS HERE THAT EXTRACT KEYWORDS FROM THE HTML CONTENT.

The Meta Info functions take a BeautifulSoup object as input and return
the extracted information as a tuple, where the first element is the key
under which the information should be stored and the second is the information.

The Keyword functions take a BeautifulSoup object as input and return
the extracted keywords as a list.

"""

import re


# +++++Keyword Getters+++++

def keyword_getter_tag(soup):
    tags = soup.find_all("meta", property='\\"article:tag\\"')

    keywords = []
    for tag in tags:
        keyword = tag.get('content')
        keyword = re.sub(r'^\\+"+|\\+"+$', '', keyword)
        keywords.append(keyword)

    return keywords


def keyword_getter_news(soup):
    """This functions gets keywords from the meta tag with the name 'news_keywords'.
    It assumes that the keywords are separated by commas or semicolons.
    This is the standard format on some websites.
    """

    separator_pattern = r",|;"
    keywords = soup.find("meta", attrs={"name": "news_keywords"})
    if keywords:
        keyword_list = re.split(
            separator_pattern, keywords["content"]
        )
        keyword_list = [
            keyword.strip() for keyword in keyword_list
        ]
        keyword_list = [
            keyword for keyword in keyword_list if len(keyword) > 0
        ]
        return keyword_list
    return []


def keyword_getter_basic(soup):
    """This functions gets keywords from the meta tag with the name 'keywords'.
    It assumes that the keywords are separated by commas or semicolons.
    This is the standard format on many websites.
    """

    separator_pattern = r",|;"
    keywords = soup.find("meta", attrs={"name": "keywords"})
    if keywords:
        keyword_list = re.split(
            separator_pattern, keywords["content"]
        )
        keyword_list = [
            keyword.strip() for keyword in keyword_list
        ]
        keyword_list = [
            keyword for keyword in keyword_list if len(keyword) > 0
        ]
        return keyword_list
    return []


def no_keywords(_):
    """This function is a placeholder for when a website has no keywords."""

    return []


# +++++Other Info Getters+++++

def get_h1(soup):
    """This function extracts the text from the first h1 tag
    (often, this is the main header) in the HTML content.
    """

    h1 = soup.find("h1")
    if h1:
        return 'h1', h1.get_text(strip=True)
    return 'h1', None


def get_h2(soup):
    """This function extracts the text from the first h2 tag
    (often, this is the subheader) in the HTML content.
    """

    h2 = soup.find("h2")
    if h2:
        return 'h2', h2.get_text(strip=True)
    return 'h2', None


def get_image_links(soup):
    """This function extracts the URLs of all images in the HTML content.
    """

    image_links = []
    for img in soup.find_all("img"):
        image_links.append(img["src"])
    return 'image_links', image_links
