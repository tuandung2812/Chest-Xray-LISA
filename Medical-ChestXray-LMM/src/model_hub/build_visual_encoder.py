from transformers import CLIPImageProcessor


def build_visual_encoder(config):
    cfg_visual_encoder = config.model.visual_encoder
    visual_encoder = CLIPImageProcessor.from_pretrained(cfg_visual_encoder.hf_visual_model_id)
    return visual_encoder
