import os
import time
import shutil
import tqdm

import wandb
import torch

from training_hub.monitor.monitor import (
    AverageMeter, 
    ProgressMeter, 
    dict_to_cuda, 
    intersectionAndUnionGPU,
    Summary
)


class LISATrainer:
    def __init__(
        self,
        config,
        dataloader_train,
        dataloader_val,
        tokenizer,
        model,
        scheduler,
        writer
    ):
        self.cfg = config
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.tokenizer = tokenizer
        self.model = model
        self.scheduler = scheduler
        self.writer = writer

        if not self.cfg.monitor.disable_wandb:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            wandb.init(project=self.cfg.monitor.project_name, 
                       name=self.cfg.monitor.run_name)
            conversation_table = wandb.Table(columns=["question/answer"]) 
            wandb.log({"Conversation": conversation_table})

    def train_epoch(self, epoch, train_iter):
        """Main training loop."""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        ce_losses = AverageMeter("CeLoss", ":.4f")
        mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
        mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
        mask_losses = AverageMeter("MaskLoss", ":.4f")

        progress = ProgressMeter(
            self.cfg.train.steps_per_epoch,
            [
                batch_time,
                losses,
                ce_losses,
                mask_losses,
                mask_bce_losses,
                mask_dice_losses,
            ],
            prefix="Epoch: [{}]".format(epoch),
        )

        self.model.train()
        end = time.time()
        for global_step, input_dict in enumerate(self.dataloader_train):
            if global_step >  self.cfg.train.steps_per_epoch:
                break
            else:
                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                if self.cfg.model.lmm.precision == "fp16":
                    input_dict["images"] = input_dict["images"].half()
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif self.cfg.model.lmm.precision == "bf16":
                    input_dict["images"] = input_dict["images"].bfloat16()
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    input_dict["images"] = input_dict["images"].float()
                    input_dict["images_clip"] = input_dict["images_clip"].float()

                output_dict = self.model(**input_dict)

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                mask_bce_loss = output_dict["mask_bce_loss"]
                mask_dice_loss = output_dict["mask_dice_loss"]
                mask_loss = output_dict["mask_loss"]

                losses.update(loss.item(), input_dict["images"].size(0))
                ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
                self.model.backward(loss)
                self.model.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if self.cfg.train.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()

                    losses.all_reduce()
                    ce_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()

                if not self.cfg.monitor.disable_wandb:
                    wandb.log({'train/loss': losses.avg,'train/ce_loss' :ce_losses.avg, 'train/mask_bce_loss' : mask_bce_losses.avg, 'train/mask_dice_loss': mask_dice_losses.avg, 'train/mask_loss': mask_losses.avg})
                else:
                    if global_step % 100 == 0:
                        print(f'Epoch [{epoch}] || Train_loss: {losses.avg:.3f} || CE loss: {ce_losses.avg:.3f} || Mask BCE: {mask_bce_losses.avg:.3f} || Dice: {mask_dice_losses.avg:.3f}')
                if self.cfg.infrastucture.local_rank == 0:
                    progress.display(global_step + 1)
                    self.writer.add_scalar("train/loss", losses.avg, global_step)
                    self.writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)

                    self.writer.add_scalar(
                        "train/mask_bce_loss", mask_bce_losses.avg, global_step
                    )
                    self.writer.add_scalar(
                        "train/mask_dice_loss", mask_dice_losses.avg, global_step
                    )
                    self.writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)

                    self.writer.add_scalar(
                        "metrics/total_secs_per_batch", batch_time.avg, global_step
                    )
                    self.writer.add_scalar(
                        "metrics/data_secs_per_batch", data_time.avg, global_step
                    )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        return train_iter

    def train(self):
        best_score, cur_ciou = 0.0, 0.0
        train_iter = iter(self.dataloader_train)
        for epoch in tqdm.tqdm(range(self.cfg.train.epoch.start, self.cfg.train.epoch.end)):
            train_iter = self.train_epoch(epoch, train_iter)

            giou, ciou = validate(self.dataloader_val, self.model, epoch, self.writer, self.cfg, self.tokenizer)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

            save_dir = self.cfg.train.dirpath_ckpt
            if self.cfg.infrastucture.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        self.cfg.train.dirpath_log,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            self.model.save_checkpoint(save_dir)


def validate(val_loader, model, epoch, writer, cfg, tokenizer):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    ce_losses = AverageMeter("CELoss", ":6.3f", Summary.SUM)

    model.eval()

    data_for_wb = []
    for idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        if idx == cfg.debug.steps_per_val:
            break
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if cfg.model.lmm.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif cfg.model.lmm.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        output_ids = output_dict['output_ids']
        ce_loss = output_dict['ce_loss']
        # print('ce_loss: ', ce_loss)
        # print('output_ids shape: ',output_ids.shape)
        ce_losses.update(ce_loss.item(), input_dict["images"].size(0))

        text = tokenizer.decode(output_ids)
        
        truncated_text = text
        if not cfg.monitor.disable_wandb and (idx % 20 == 0):
            # print(input_dict["conversation_list"])
            wandb.log({'val/examples': [wandb.Image(input_dict["images"][0].permute(1, 2, 0).float().cpu().numpy(),caption = 'image'), 
                                    wandb.Image(masks_list[0].cpu().numpy()*255,caption = 'label'), 
                                    wandb.Image(output_list[0].cpu().numpy()*255, caption = "pred")
                                    ]})
            # conversation_table = wandb.Table(columns=["question/answer","generated"], data=[input_dict["conversation_list"], truncated_text])
            data_for_wb.append([idx, input_dict["conversation_list"][0], truncated_text])
            conversation_table = wandb.Table(columns=["idx","conversation","predicted_caption"], data =data_for_wb)
            # conversation_table.add_data(input_dict["conversation_list"], truncated_text)
            # print('conv: ', conversation_table)
            wandb.log({"Conversation": conversation_table}) 
            wandb.log({'val/ce_loss' :ce_losses.avg})
        elif idx % 20 == 0:
            print(f"Index [{idx}] || Conversation: {input_dict['conversation_list'][0]}")
            print(f"Predicted Caption: {truncated_text}")
            print(f"CE Loss: {ce_losses.avg:.3f}")

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
        if idx % 20 == 0:
            print(f"IoU: {acc_iou}")

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if cfg.infrastucture.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        # print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return giou, ciou