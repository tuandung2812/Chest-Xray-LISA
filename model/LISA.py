from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

# from .llava.model.language_model.llava_mistral import (LlavaMistralForCausalLM,
#                                                      LlavaMistralModel)

# from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
#                                                      LlavaLlamaModel)
from .segment_anything import build_sam_vit_h , build_sam_vit_b
from .llava.model.language_model.llava_mistral import (LlavaMistralForCausalLM,
                                                     LlavaMistralModel)
# from loss_fn import caculate_loss_att_fixed_cn

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        try:
        # self.visual_model = build_sam_vit_h(self.vision_pretrained)\
            self.visual_model = build_sam_vit_b(self.vision_pretrained)
        except:
            self.visual_model = build_sam_vit_h(self.vision_pretrained)\

        for param in self.visual_model.parameters():
            param.requires_grad = False
        # if config.train_mask_decoder:
        self.visual_model.mask_decoder.train()
        for param in self.visual_model.mask_decoder.parameters():
            param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaMistralModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

class LISAForCausalLM(LlavaMistralForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        # Obtain the image embeddings using the SAM image encoder
        # images DIM = (B, 3C, 1024, 1024)
        image_embeddings = self.get_visual_embs(images) # DIM [B, 256, 64, 64]
        # print('image_embeddings: ', image_embeddings.shape)
        batch_size = image_embeddings.shape[0]
        # print('offset: ', offset)
        assert batch_size == len(offset) - 1    

        # Define the seg token
        # print('input_ids: ', input_ids.shape)
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        # print('seg_token_mask: ', seg_token_mask.shape, seg_token_mask)
        # Fix later
        try:
              seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
        except:
              seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool(),
                ],
                dim=1,
            )
        # print('seg_token_mask 2: ', seg_token_mask.shape, seg_token_mask)

        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        try:
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
                dim=1,
            )
        except:
              seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool(),
                ],
                dim=1,
            )

        # print('seg_token_mask 3: ', seg_token_mask.shape, seg_token_mask)

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            output_logits=  []
            output_ce_losses = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                # print(start_i, end_i)
                # inference using llava
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    labels = labels[start_i:end_i],
                    output_hidden_states=True,
                )
                # Getting the output as hidden_states.
                # For llava mistral, we manually need to get the final hidden states
                output_hidden_states.append(output_i.hidden_states[-1])
                output_logits.append(output_i.logits)
                output_ce_losses.append(output_i.loss.item())
                print('output_i ',output_i.loss)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
            output_ce_losses = torch.Tensor(output_ce_losses)
            print('CE Loss: ',output_ce_losses, output_ce_losses.shape)

            output_logits_list = []
            output_logits_level = torch.cat(output_logits, dim=0)
            output_logits = output_logits_level[0]
            # print(output_logits.shape)
            output_ids  = torch.argmax(output_logits, dim=-1)
            # print('output_ids: ', output_ids.shape, output_ids)
            output_ids = output_ids.unsqueeze(0)

            filtered_output = torch.cat([torch.zeros((output_ids.shape[0], 255)).bool().cuda(),
                                            (labels[:,1:] != -100).int(),
                                            torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()],dim=1)
            output_ids = output_ids[filtered_output == 1]
            output = None
        else:
            images_clip_list = []
            attention_weights=  []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            # Output obtain from llava
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                output_attentions=True  # Enable attention output
            )
            output_hidden_states = output.hidden_states
            attention_weights.append(output_i.attentions)
        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]
        pred_embeddings_ = []
        # print('seg_token_offset: ', seg_token_offset.shape)
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            # print('offset product: ', pred_embeddings[start_i:end_i].shape)
            # print('start i end i: ', start_i, end_i)
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        # Passing the text embedding to the text prompt
        # The length of pred_embeddings is the batch size.
        # Each element in pred_embedding is [3,256]
        for i in range(len(pred_embeddings)):
            # print('i: ', i)
            # print('text_embeds: ',pred_embeddings[i].unsqueeze(1).shape)
            (
                # Sparse embeddings D = {3,1,256} Based on text_embeds
                # Dense embeddings D = {3,256,64,64} Based on a no-mask embeds
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            # We use the image embeddings
            # print('sparse_embeddings: ', sparse_embeddings.shape)
            # print('dense_embeddings: ', dense_embeddings.shape)
            # print('image_embeddings: ', image_embeddings[i].unsqueeze(0).shape)

            # image_embeddings = # DIM [B, 256, 64, 64]
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            # print('low_res_masks: ', low_res_masks.shape)
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
            # print('len predmask: ',len(pred_masks))
            # print(pred_mask.shape)
        model_output = output
        gt_masks = masks_list

        if inference:
    
            return {
                "output_ids": output_ids,
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                'ce_loss': output_ce_losses
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                'output_ids': output_ids,
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,

            }

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "attention_weights": attention_weights  # Include attention weights
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,

            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks

# class LISAForCausalLM(LlavaMistralForCausalLM):
#     def __init__(
#         self,
#         config,
#         **kwargs,
#     ):
#         if not hasattr(config, "train_mask_decoder"):
#             config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
#             config.mm_vision_tower = kwargs.get(
#                 "vision_tower", "openai/clip-vit-large-patch14"
#             )
#             self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
#             self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
#             self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
#         else:
#             config.mm_vision_tower = config.vision_tower
            
#         self.seg_token_idx = kwargs.pop("seg_token_idx")

#         super().__init__(config)

#         self.model = LisaModel(config, **kwargs)

#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_visual_embs(self, pixel_values: torch.FloatTensor):
#         with torch.no_grad():
#             image_embeddings_list = []
#             for i in range(pixel_values.shape[0]):
#                 torch.cuda.empty_cache()
#                 image_embeddings = self.model.visual_model.image_encoder(
#                     pixel_values[i].unsqueeze(0)
#                 )
#                 image_embeddings_list.append(image_embeddings)
#             torch.cuda.empty_cache()
#             image_embeddings = torch.cat(image_embeddings_list, 0)
#             return image_embeddings

#     def forward(self, **kwargs):
#         if "past_key_values" in kwargs:
#             return super().forward(**kwargs)
#         return self.model_forward(**kwargs)

#     def model_forward(
#         self,
#         images: torch.FloatTensor,
#         images_clip: torch.FloatTensor,
#         input_ids: torch.LongTensor,
#         labels: torch.LongTensor,
#         attention_masks: torch.LongTensor,
#         offset: torch.LongTensor,
#         masks_list: List[torch.FloatTensor],
#         label_list: List[torch.Tensor],
#         resize_list: List[tuple],
#         inference: bool = False,
#         text_only: bool = False,
#         **kwargs,
#     ):
#         # Obtain the image embeddings using the SAM image encoder
#         # images DIM = (B, 3C, 1024, 1024)
#         # if...
#         print('model text only: ', text_only)
#         image_embeddings = self.get_visual_embs(images) # DIM [B, 256, 64, 64]
#         batch_size = image_embeddings.shape[0]
#         assert batch_size == len(offset) - 1    

#         # Define the seg token
#         seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
#         seg_token_mask = torch.cat(
#             [
#                 seg_token_mask,
#                 torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
#             ],
#             dim=1,
#         )
#         # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
#         seg_token_mask = torch.cat(
#             [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
#             dim=1,
#         )

#         if inference:
#             n_batch = 1
#             length = input_ids.shape[0]
#             assert images_clip.shape[0] == 1
#             images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

#             output_hidden_states = []
#             output_logits=  []
#             output_ce_losses = []
#             for i in range(n_batch):
#                 start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
#                 print(start_i, end_i)
#                 # inference using llava
#                 output_i = super().forward(
#                     images=images_clip_extend[: end_i - start_i],
#                     attention_mask=attention_masks[start_i:end_i],
#                     input_ids=input_ids[start_i:end_i],
#                     labels = labels[start_i:end_i],
#                     output_hidden_states=True,
#                 )
#                 # print(output_i.logits.shape)
#                 # Getting the output as hidden_states.
#                 # For llava mistral, we manually need to get the final hidden states
#                 output_hidden_states.append(output_i.hidden_states[-1])
#                 output_logits.append(output_i.logits)
#                 output_ce_losses.append(output_i.loss.item())
#                 print('output_i ',output_i.loss)
#                 torch.cuda.empty_cache()

#             output_hidden_states_list = []
#             output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
#             output_hidden_states_list.append(output_hidden_states_level)
#             output_hidden_states = output_hidden_states_list

#             # output_ce_losses_list = []
#             # output_ce_losses_level = torch.cat(output_ce_losses, dim=0)
#             # output_ce_losses_list.append(output_ce_losses_level)
#             # output_ce_losses = output_ce_losses_list
#             output_ce_losses = torch.Tensor(output_ce_losses)
#             print('CE Loss: ',output_ce_losses, output_ce_losses.shape)

#             output_logits_list = []
#             output_logits_level = torch.cat(output_logits, dim=0)
#             output_logits = output_logits_level[0]
#             # print(output_logits.shape)
#             output_ids  = torch.argmax(output_logits, dim=-1)
#             # print('output_ids: ', output_ids.shape, output_ids)
#             output = None

#         else:
#             images_clip_list = []
#             for i in range(len(offset) - 1):
#                 start_i, end_i = offset[i], offset[i + 1]
#                 images_clip_i = (
#                     images_clip[i]
#                     .unsqueeze(0)
#                     .expand(end_i - start_i, -1, -1, -1)
#                     .contiguous()
#                 )
#                 images_clip_list.append(images_clip_i)
#             images_clip = torch.cat(images_clip_list, dim=0)

#             # Output obtain from llava
#             output = super().forward(
#                 images=images_clip,
#                 attention_mask=attention_masks,
#                 input_ids=input_ids,
#                 labels=labels,
#                 output_hidden_states=True,
#             )
#             output_hidden_states = output.hidden_states
        
#         if text_only[0]:
#             print("entered text_only")
#             if inference:
#                 return {
#                     'ce_loss': output_ce_losses,
#                     'output_logits': output_logits,
#                     'output_ids': output_ids
#                 }
#             else:
#                 model_output = output
#                 ce_loss = model_output.loss
#                 # ce_loss = ce_loss * self.ce_loss_weight
#                 loss = ce_loss

#                 return {
#                     "loss": loss,
#                     "ce_loss": ce_loss}

#         hidden_states = []

#         assert len(self.model.text_hidden_fcs) == 1
#         hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
#         print('got past text only')
#         last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
#         print('last_hidden_state: ', last_hidden_state.shape)
#         print('seg_token_mask: ', seg_token_mask.shape, seg_token_mask)
#         pred_embeddings = last_hidden_state[seg_token_mask]
#         print('pred_embeddings from last_hidden_state: ', pred_embeddings.shape)
#         seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
#         print('seg_token_counts: ', seg_token_counts)
#         seg_token_offset = seg_token_counts.cumsum(-1)
#         print('seg_token_offset: ', seg_token_offset, seg_token_offset.shape)

#         seg_token_offset = torch.cat(
#             [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
#         )
#         print('seg_token_offset 2: ', seg_token_offset, seg_token_offset.shape)

#         seg_token_offset = seg_token_offset[offset]
#         print('offset: ',offset)

#         print('seg_token_offset 3: ', seg_token_offset, seg_token_offset.shape)

#         pred_embeddings_ = []
#         for i in range(len(seg_token_offset) - 1):
#             print('i: ', i)
#             start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
#             pred_embeddings_.append(pred_embeddings[start_i:end_i])
#         pred_embeddings = pred_embeddings_

#         multimask_output = False
#         pred_masks = []
#         # Passing the text embedding to the text prompt
#         # The length of pred_embeddings is the batch size.
#         # Each element in pred_embedding is [3,256]
#         print('len(pred_embeddings): ', len(pred_embeddings))
#         for i in range(len(pred_embeddings)):
#             (
#                 # Sparse embeddings D = {3,1,256} Based on text_embeds
#                 # Dense embeddings D = {3,256,64,64} Based on a no-mask embeds
#                 sparse_embeddings,
#                 dense_embeddings,
#             ) = self.model.visual_model.prompt_encoder(
#                 points=None,
#                 boxes=None,
#                 masks=None,
#                 text_embeds=pred_embeddings[i].unsqueeze(1),
#             )
#             sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
#             # We use the image embeddings
#             # image_embeddings = # DIM [B, 256, 64, 64]
#             low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
#                 image_embeddings=image_embeddings[i].unsqueeze(0),
#                 image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=multimask_output,
#             )
#             pred_mask = self.model.visual_model.postprocess_masks(
#                 low_res_masks,
#                 input_size=resize_list[i],
#                 original_size=label_list[i].shape,
#             )
#             pred_masks.append(pred_mask[:, 0])

#         model_output = output
#         gt_masks = masks_list
        
#         # output_ids  = torch.argmax(output_logits, dim=-1)
#         # print('output_ids: ', output_ids.shape, output_ids)
#         if inference:
#             return {
#                 "pred_masks": pred_masks,
#                 "gt_masks": gt_masks,
#                 'output_ids': output_ids
#             }

#         # output = model_output.logits

#         ce_loss = model_output.loss
#         ce_loss = ce_loss * self.ce_loss_weight
#         mask_bce_loss = 0
#         mask_dice_loss = 0
#         num_masks = 0
#         for batch_idx in range(len(pred_masks)):
#             print('batch_idx: ', batch_idx)
#             gt_mask = gt_masks[batch_idx]
#             pred_mask = pred_masks[batch_idx]
            
#             # print(gt_mask, pred_mask)

#             assert (
#                 gt_mask.shape[0] == pred_mask.shape[0]
#             ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
#                 gt_mask.shape, pred_mask.shape
#             )
#             mask_bce_loss += (
#                 sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
#                 * gt_mask.shape[0]
#             )
#             mask_dice_loss += (
#                 dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
#                 * gt_mask.shape[0]
#             )
#             num_masks += gt_mask.shape[0]

#         mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
#         mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
#         mask_loss = mask_bce_loss + mask_dice_loss

#         loss = ce_loss + mask_loss
        
#         if inference:
#             return {
#                 "pred_masks": pred_masks,
#                 "gt_masks": gt_masks,
#             }


#         return {
#             "loss": loss,
#             "ce_loss": ce_loss,
#             "mask_bce_loss": mask_bce_loss,
#             "mask_dice_loss": mask_dice_loss,
#             "mask_loss": mask_loss,
#         }

#     def evaluate(
#         self,
#         images_clip,
#         images,
#         input_ids,
#         resize_list,
#         original_size_list,
#         max_new_tokens=32,
#         tokenizer=None,
#     ):
#         with torch.no_grad():
#             outputs = self.generate(
#                 images=images_clip,
#                 input_ids=input_ids,
#                 max_new_tokens=max_new_tokens,
#                 num_beams=1,
#                 output_hidden_states=True,
#                 return_dict_in_generate=True,
#             )
#             output_hidden_states = outputs.hidden_states[-1]
#             output_ids = outputs.sequences
        
#             seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
#             # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
#             seg_token_mask = torch.cat(
#                 [
#                     torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
#                     seg_token_mask,
#                 ],
#                 dim=1,
#             )

#             hidden_states = []

#             assert len(self.model.text_hidden_fcs) == 1
#             hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

#             last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
#             pred_embeddings = last_hidden_state[seg_token_mask]

#             seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
#             seg_token_offset = seg_token_counts.cumsum(-1)
#             seg_token_offset = torch.cat(
#                 [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
#             )

#             pred_embeddings_ = []
#             for i in range(len(seg_token_offset) - 1):
#                 start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
#                 pred_embeddings_.append(pred_embeddings[start_i:end_i])
#             pred_embeddings = pred_embeddings_

#             image_embeddings = self.get_visual_embs(images)

#             multimask_output = False
#             pred_masks = []
#             for i in range(len(pred_embeddings)):
#                 (
#                     sparse_embeddings,
#                     dense_embeddings,
#                 ) = self.model.visual_model.prompt_encoder(
#                     points=None,
#                     boxes=None,
#                     masks=None,
#                     text_embeds=pred_embeddings[i].unsqueeze(1),
#                 )

#                 sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
#                 low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
#                     image_embeddings=image_embeddings[i].unsqueeze(0),
#                     image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
#                     sparse_prompt_embeddings=sparse_embeddings,
#                     dense_prompt_embeddings=dense_embeddings,
#                     multimask_output=multimask_output,
#                 )
#                 pred_mask = self.model.visual_model.postprocess_masks(
#                     low_res_masks,
#                     input_size=resize_list[i],
#                     original_size=original_size_list[i],
#                 )
#                 pred_masks.append(pred_mask[:, 0])
#         if text_only:
#             return output_ids
#         return output_ids, pred_masks
