import os
from diffusers.loaders import FromOriginalModelMixin
from transformers.models.clip import CLIPVisionModelWithProjection, CLIPImageProcessor


class OriginCLIPVisionModelWithProjection(CLIPVisionModelWithProjection, FromOriginalModelMixin):

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path_or_dict=None, **kwargs):
        model = OriginCLIPVisionModelWithProjection.from_pretrained(pretrained_model_link_or_path_or_dict)
        return model


class OriginCLIPImageProcessor(CLIPImageProcessor, FromOriginalModelMixin):

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path_or_dict=None, **kwargs):
        model = OriginCLIPImageProcessor.from_pretrained(pretrained_model_link_or_path_or_dict)
        return model


EntryClass = [OriginCLIPVisionModelWithProjection, OriginCLIPImageProcessor]
