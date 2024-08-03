import json

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class MiniCPMVTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eot_token = "<|eot_id|>"
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"

    @property
    def eos_id(self):
        return self.eos_token_id

    @property
    def bos_id(self):
        return self.bos_token_id

    @property
    def unk_id(self):
        return self.unk_token_id

    @property
    def eot_id(self):
        return self.convert_tokens_to_ids(self.eot_token)

    @property
    def im_start_id(self):
        return self.convert_tokens_to_ids(self.im_start)

    @property
    def im_end_id(self):
        return self.convert_tokens_to_ids(self.im_end)

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text
