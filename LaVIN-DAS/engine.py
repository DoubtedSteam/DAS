import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import torch.nn.functional as F


def clean_flag(model):
    for layer in model.layers:
        layer.skipped_flag = -1.

def apply_flag(model, select):
    for i, layer in enumerate(model.layers):
        if i in select:
            layer.skipped_flag = 1.

def get_prob(model):
    # return F.softmax(model.scores * 10, dim=-1)
    return torch.sigmoid(model.scores)


def tensor_in_list(tensor_list, new_tensor):
    for tensor in tensor_list:
        if torch.equal(tensor, new_tensor):
            return True
    return False


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    dataset_val=None,
                    sampler_val=None,
                    local_rank=-1,
                    world_size=-1,
                ):
  
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 300

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    prefix_img = torch.tensor(data_loader.dataset.tokenizer.encode("Image: ", bos=False, eos=False), dtype=torch.int64)
    prefix_nonimg = torch.tensor(data_loader.dataset.tokenizer.encode("Image: N/A", bos=False, eos=False), dtype=torch.int64)

    #### NAS related ####
    nas_epoch = args.nas_epoch if args.search_mode else 0
    skipped_num = args.skipped_num
    warmup_epoch = args.nas_warmup_epoch
    nas_step = 3
    nas_turn = len(data_loader) // 150
    nas_count = 0
    
    if args.search_mode:
        select = []
    else:
        if args.skip_list != '[]':
            select = [int(i) for i in args.skip_list[1:-1].split(',')]
        else:
            select = []
    apply_flag(model.module, select)
    print(select)

    for data_iter_step, (examples, labels, example_mask, images, indicators) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if data_iter_step / len(data_loader) + epoch >= nas_epoch:
            c_loss = model(examples, labels, images=images, prefix_img=prefix_img, prefix_nonimg=prefix_nonimg, img_indicators=indicators)
        else:
            clean_flag(model.module)
            prob = get_prob(model.module)
            select = torch.multinomial(prob, skipped_num)
            if local_rank != 0:
                select = torch.zeros_like(select)
            torch.distributed.all_reduce(select)
            apply_flag(model.module, select)
            nas_count += 1

            c_loss = model(examples, labels, images=images, prefix_img=prefix_img, prefix_nonimg=prefix_nonimg, img_indicators=indicators)

            if data_iter_step % 300 == 0:
                print(prob)
                select = torch.sort(prob)[1]
                select = select[-skipped_num:]
                print(select)
        
            if nas_count >= nas_turn and data_iter_step / len(data_loader) + epoch >= warmup_epoch:
                nas_count = 0
                val_loader = torch.utils.data.DataLoader(
                    dataset_val, sampler=sampler_val,
                    batch_size=args.batch_size * 4,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                )
                val_loader.sampler.set_epoch(data_iter_step // nas_turn)

                val_examples, val_labels, _, val_images, val_indicators = next(iter(val_loader))

                selects = []
                t_losses = []

                prob = get_prob(model.module)

                for k in range(nas_step):
                    clean_flag(model.module)
                    select = torch.sort(torch.multinomial(prob, skipped_num))[0]
                    if local_rank != 0:
                        select = torch.zeros_like(select)
                    else:
                        while tensor_in_list(selects, select):
                            select = torch.sort(torch.multinomial(prob, skipped_num))[0]
                    torch.distributed.all_reduce(select)
                    selects.append(select)

                    apply_flag(model.module, select)

                    model.eval()
                    with torch.no_grad():
                        t_loss = model(
                            val_examples, 
                            val_labels, 
                            images=val_images, 
                            prefix_img=prefix_img, 
                            prefix_nonimg=prefix_nonimg, 
                            img_indicators=val_indicators
                        )
                    model.train()
                    t_losses.append(t_loss.item())

                rewards = []
                for k in range(nas_step):
                    rewards.append(math.exp(-t_losses[k]))
                rewardb = sum(rewards) / nas_step
                
                misc.all_reduce_mean(rewardb)

                lr = 0.05
                txt_len = len(model.module.layers)
                prob.cuda()
                for k in range(nas_step):
                    for i in selects[k]:
                        i = i.item()
                        model.module.scores.data[i] += (rewards[k] - rewardb) * prob[i] * (1 - prob[i]) * lr

                torch.distributed.all_reduce(model.module.scores.data)
                model.module.scores.data = model.module.scores.data / world_size

        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        if torch.isnan(loss):
            print("NaN loss encountered. Skipping this batch.")
            continue

        loss = loss/accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=args.clip_grad)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        with torch.no_grad():
             c_loss  = model(examples, labels)
        loss = c_loss
        loss_value = loss.item()

        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
