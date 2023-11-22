import torch
from torch import nn

from functools import partial

DEFAULT_PLACEHOLDER_TOKEN = ["*"]


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, \
        f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, \
        f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]

# consider a more 'hard-coded' approach, so you can get it working with fewer changes
class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            placeholder_modules=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):

        # Module Embedding Manager does not support various operations and should fail fast (and loud)
        assert initializer_words is None, "unsupported"
        assert per_image_tokens is False, "unsupported"
        assert num_vectors_per_token == 1, "unsupported"
        assert progressive_words is False, "unsupported"

        assert len(placeholder_strings) == len(placeholder_modules), \
            "all strings must have a module to generate embedding"

        super().__init__()

        self.string_to_token_dict = {}
        self.string_to_module_dict = nn.ModuleDict()

        if hasattr(embedder, 'tokenizer'):  # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            # token_dim = 768

        else:  # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            # token_dim = 1280

        for idx, placeholder_string in enumerate(placeholder_strings):

            self.string_to_token_dict[placeholder_string] = get_token_for_string(placeholder_string)
            self.string_to_module_dict[placeholder_string] = placeholder_modules[idx]

    def forward(
            self,
            tokenized_text,
            embedded_text,
            special_embeddings
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():

            placeholder_module = self.string_to_module_dict[placeholder_string].to(device)

            placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
            embedded_text[placeholder_idx] = placeholder_module(special_embeddings)

        return embedded_text

    def save(self, ckpt_path):
        assert False, "unsupported"
        # torch.save({"string_to_token": self.string_to_token_dict,
        #             "string_to_param": self.string_to_module_dict}, ckpt_path)

    def load(self, ckpt_path):
        assert False, "unsupported"
        # ckpt = torch.load(ckpt_path, map_location='cpu')

        # self.string_to_token_dict = ckpt["string_to_token"]
        # self.string_to_module_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        assert False, "unsupported"
        # all_params = torch.cat(list(self.string_to_module_dict.values()), axis=0)  # num_placeholders x embedding_dim
        # param_norm_squared = (all_params * all_params).sum(axis=-1)               # num_placeholders

        # return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_module_dict.parameters()

    def embedding_to_coarse_loss(self):
        assert False, "unsupported"

        # loss = 0.
        # num_embeddings = len(self.initial_embeddings)

        # for key in self.initial_embeddings:
        #     optimized = self.string_to_module_dict[key]
        #     coarse = self.initial_embeddings[key].clone().to(optimized.device)

        #     loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        # return loss


class SimpleEmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):

        # Module Embedding Manager does not support various operations and should fail fast (and loud)
        assert initializer_words is None, "unsupported"
        assert per_image_tokens is False, "unsupported"
        assert num_vectors_per_token == 1, "unsupported"
        assert progressive_words is False, "unsupported"

        super().__init__()

        self.string_to_token_dict = {}
        self.string_to_param_dict = nn.ParameterDict()

        if hasattr(embedder, 'tokenizer'):  # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            token_dim = 768

        else:  # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            token_dim = 1280

        for idx, placeholder_string in enumerate(placeholder_strings):

            token = get_token_for_string(placeholder_string)

            token_params = torch.nn.Parameter(torch.rand(size=(1, token_dim), requires_grad=True))

            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params

    def forward(
            self,
            tokenized_text,
            embedded_text,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():

            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)

            placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
            embedded_text[placeholder_idx] = placeholder_embedding  # change to using modules

        return embedded_text

    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict}, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):

        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss
