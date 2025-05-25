"""
This file contains functions that are used to check if an html element
has so many duplicates in the corpus that it should be removed to not
overrepresent in the data.

All functions should have the following signature:
    def function_name(
        name: str,
        text: str,
        count: int
    ) -> bool

They return False if the element should be kept and True if it should
be removed.
"""

import re



def spektrum_checker(name, text, count):
    if name == 'p' and count > 3:
        return True
    return False


def zeit_checker(name, text, count):
    if name in {'span', 'td',}:
        return True
    if name == 'p' and text.count(' ') > 10 and count > 2:
        return True
    if name in {'h1', 'h2', 'h3', 'p'} and count > 5:
        return True
    return False
