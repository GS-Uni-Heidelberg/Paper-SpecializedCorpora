import re


def zeit_dpa(data):
    text = data['text']
    if re.search(r'© dpa-infocom', text):
        return False
    return True


def zeit_paywall(data):
    tier = data.get('article:content_tier')
    if tier == 'locked':
        return False
    return True


def zeit_multipage(data):
    url = data['url']
    if re.search(r'/seite-[0-9]+', url):
        return False
    return True
