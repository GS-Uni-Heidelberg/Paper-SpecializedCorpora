from . import _meta_functions


class MetaRetriever():
    """Class for extracting metadata BeautifulSoup objects."""

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if value is not None:
            try:
                iter(value)
            except TypeError as excep:
                message = "Labels must be an iterable."
                raise ValueError(message) from excep
        self._labels = value

    @property
    def kywd_extraction_func(self):
        return self._kywd_extraction_func

    @kywd_extraction_func.setter
    def kywd_extraction_func(self, value):
        self._kywd_extraction_func = value

    @property
    def other_info_funcs(self):
        return self._other_info_funcs

    @other_info_funcs.setter
    def other_info_funcs(self, value):
        try:
            iter(value)
        except TypeError as excep:
            raise ValueError(
                "Other info functions must be an iterable.") from excep
        self._other_info_funcs = value

    def __init__(
        self, metadata_labels_keep=None,
        kywd_extraction_func='keyword_getter_basic',
        other_info_funcs=('get_h1')
    ):
        self._labels = metadata_labels_keep
        self._kywd_extraction_func = kywd_extraction_func
        self._other_info_funcs = other_info_funcs

    def fill_meta_dict(self, soup):
        meta_data = soup.find_all("meta")
        meta_dict = {}
        for datum in meta_data:
            name = datum.get("name")
            if name:
                meta_dict[name] = datum.get("content")
            else:
                proprty = datum.get("property")
                if proprty:
                    meta_dict[proprty] = datum.get("content")

        meta_dict = {
            k: v for k, v in meta_dict.items()
            if k in self.labels
        }

        return meta_dict

    def fill_other_info(self, soup):
        other_info = {}
        for function in self.other_info_funcs:
            function = getattr(_meta_functions, function)
            key, value = function(soup)
            other_info[key] = value

        return other_info

    def full_pipeline(self, soup):
        meta_dict = self.fill_meta_dict(soup)
        other_info = self.fill_other_info(soup)
        if self.kywd_extraction_func:
            kywd_extraction_func = getattr(
                _meta_functions, self.kywd_extraction_func
            )
            keywords = kywd_extraction_func(soup)
            meta_dict["keywords"] = keywords
        meta_dict.update(other_info)
        return meta_dict
