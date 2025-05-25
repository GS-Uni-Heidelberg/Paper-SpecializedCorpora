import yaml
import string
import re


class SysPrompt():
    """Class that represents a system prompt and associated metadata."""

    def __init__(self):
        self.name = ""
        self.sysprompt = ""
        self.language = ""
        self.nshot = 0

    def load_from_file(self, prompt_file, name):

        with open(prompt_file, "r", encoding='utf-8') as f:
            prompts = yaml.safe_load(f)

        self.name = name
        self.sysprompt = prompts[name].get("sysprompt", "")
        self.language = prompts[name].get("language", "")
        self.nshot = prompts[name].get("nshot", 0)

    def set_prompt(self, name, sysprompt, language, nshot):
        self.name = name
        self.sysprompt = sysprompt
        self.language = language
        self.nshot = nshot

    def get_sysprompt(self):
        return self.sysprompt

    def get_userprompt(self, user_content):
        return user_content

    def __str__(self):
        return (
            f"Name: {self.name}\n\n"
            f"Sysprompt: {self.sysprompt}\n"
        )


class SysUserPrompt(SysPrompt):
    """Class that represents a set of prompts and associated metadata.

    The prompts represented by this class should be split into a system prompt
    and a user prompt.
    The user prompt should be split into a beginning and an
    end. In between the beginning and the end, functions will automatically
    insert a text input that should be annotated.
    """

    @property
    def userprompt_begin(self):
        return self._userprompt_begin

    @userprompt_begin.setter
    def userprompt_begin(self, value):
        if not isinstance(value, str):
            print("Warning: userprompt_begin should be a string.")
            return
        self._userprompt_begin = value

    @property
    def userprompt_end(self):
        return self._userprompt_end

    @userprompt_end.setter
    def userprompt_end(self, value):
        if not isinstance(value, str):
            print("Warning: userprompt_end should be a string.")
            return
        self._userprompt_end = value

    def __init__(self):
        super().__init__()
        self._userprompt_begin = ""
        self._userprompt_end = ""

    def load_from_file(self, prompt_file, name):

        with open(prompt_file, "r", encoding='utf-8') as f:
            prompts = yaml.safe_load(f)

        self.name = name
        self.sysprompt = prompts[name].get("sysprompt", "")
        self.userprompt_begin = prompts[name].get("userprompt_begin", "")
        self.userprompt_end = prompts[name].get("userprompt_end", "")
        self.language = prompts[name].get("language", "")
        self.nshot = prompts[name].get("nshot", 0)

    def set_prompt(
        self, name, sysprompt,
        userprompt_begin, userprompt_end,
        language, nshot
    ):
        self.name = name
        self.sysprompt = sysprompt
        self.language = language
        self.nshot = nshot
        self.userprompt_begin = userprompt_begin
        if self.userprompt_begin[-1] not in string.whitespace:
            self.userprompt_begin += "\n"
            print(
                "No whitespace at the end of userprompt_begin. "
                "Adding one for you."
            )
        self.userprompt_end = userprompt_end
        if (
            self.userprompt_end[0] not in string.whitespace
            and len(self.userprompt_end) > 0
        ):
            self.userprompt_end += "\n"
            print(
                "No whitespace at the beginning of userprompt_end. "
                "Adding one for you."
            )

    def get_userprompt(self, user_content):
        userprompt = (
            self.userprompt_begin
            + user_content
            + self.userprompt_end
        )
        return userprompt

    def __str__(self):
        return (
            f"Name: {self.name}\n\n"
            f"Sysprompt: {self.sysprompt}\n"
            f"Userprompt Begin: {self.userprompt_begin}\n"
            "<TEXT HERE>\n"
            f"Userprompt End: {self.userprompt_end}\n"
        )


class NeighboursPrompt(SysUserPrompt):
    """Class that represents a set of prompts and associated metadata.

    The prompts represented by this class should be split into a system prompt
    and a user prompt.
    The user prompt should be split into a beginning and an
    end. In between the beginning and the end, functions will automatically
    insert a text input that should be annotated.
    """

    def __init__(self):
        super().__init__()
        self._userprompt_begin = ""
        self._userprompt_end = ""

    def get_userprompt(self, user_content, examples):
        # Replace placeholder in userprompt_begin with user_content

        userprompt_begin = self.userprompt_begin
        userprompt_begin = re.sub(
            r"\{\{.*?\}\}",
            examples,
            userprompt_begin
        )
        if userprompt_begin == self.userprompt_begin:
            print(
                "No placeholder found in userprompt_begin. "
                "Using the original one."
                "This might be a problem."
            )
        print(user_content)
        userprompt = ''.join([
            userprompt_begin,
            user_content,
            self.userprompt_end
        ])
        return userprompt

    def __str__(self):
        return (
            f"Name: {self.name}\n\n"
            f"Sysprompt: {self.sysprompt}\n"
            f"Userprompt Begin: {self.userprompt_begin}\n"
            "<TEXT HERE>\n"
            f"Userprompt End: {self.userprompt_end}\n"
        )
