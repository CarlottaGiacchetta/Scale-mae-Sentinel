import math, sys, time, contextlib
from typing import Iterable
import torch
import util.lr_sched as lr_sched
import util.misc as misc
from lib.transforms import get_inputs_outputs  # se serve altrove
import os
import numpy as np

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,                      # NativeScalerWithGradNormCount stile MAE
    log_writer=None,
    args=None,
    scheduler=None,
    source_size_scheduler=None,
    fix_resolution_scheduler=None,
    gpu_transforms=None,              # ⬅️ NON sovrascriverla
):
    gpu_transforms=None
    """
    Logga tempi: data_wait, h2d, gpu_aug, fwd, bwd+step, zero, iter
    e riduce la comunicazione DDP grazie a no_sync() con grad accumulation.
    """
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    print_freq   = getattr(args, "print_freq", 20)
    accum_iter   = max(1, getattr(args, "accum_iter", 1))
    clip_grad    = getattr(args, "clip_grad", None)
    log_every_it = getattr(args, "log_every", 50)

    # zero_grad efficiente
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    # per chiamare metodi custom anche in DDP
    mm = model.module if hasattr(model, "module") else model
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    totals = {k: 0.0 for k in ["data","h2d","gpu_aug","fwd","bwd","zero","iter"]}
    iters = 0
    end = time.time()

    for data_iter_step, ((samples, res, targets, target_res), metadata) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        '''  
        
        for data_iter_step, batch in enumerate(data_loader):
        start = data_iter_step * args.batch_size
        end = start + args.batch_size
        if start == 426752:
            aa = True
        if aa:
            (samples, res, targets, target_res), metadata = batch
            arr = samples.detach().numpy()


            
            out_path = os.path.join("/leonardo_scratch/fast/IscrC_UMC/fmoWSentinel/preprocessed2", f"preprocessed_{start}_{end}.npy")
            np.save(out_path, arr)

            print(f"[OK] Salvato {out_path}  | shape={arr.shape}  ")
        
        ''' 
        # ===== 1) attesa dataloader =====
        data_wait = time.time() - end
        totals["data"] += data_wait
        iter_start = time.time()

        # LR step per-iter
        if data_iter_step % accum_iter == 0 and scheduler is not None:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # ===== 2) H2D =====
        t0 = time.time()
        samples    = samples.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last).float()
        res        = res.to(device, non_blocking=True).float()
        if torch.is_tensor(targets):
            targets   = targets.to(device, non_blocking=True).float()
        target_res = target_res.to(device, non_blocking=True).float()
        totals["h2d"] += (time.time() - t0)


        # ===== 4) Forward + Backward (con no_sync ai micro-step) =====
        update_now = ((data_iter_step + 1) % accum_iter == 0)

        # imposta size/flag del modello se i scheduler ci sono
        if scheduler is not None:
            target_size = scheduler.get_target_size(epoch)
            mm.set_target_size(target_size)
        if source_size_scheduler is not None:
            source_size = source_size_scheduler.get_target_size(epoch)[0]
        else:
            source_size = None
        if fix_resolution_scheduler is not None:
            fix_decoding_size = fix_resolution_scheduler.get_target_size(epoch)
            mm.set_fix_decoding_size(fix_decoding_size)

        # in BF16 su GPU
        cm = (model.no_sync() if (is_ddp and not update_now) else contextlib.nullcontext())
        with cm:
            # ----- forward -----
            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, y, mask, mean, var, pos_emb, pos_emb_decoder, samples = mm(
                    samples,
                    input_res=res,
                    targets=targets,
                    target_res=target_res,
                    mask_ratio=args.mask_ratio,
                    source_size=source_size,
                )
            # sincronizza SOLO per misurare il forward
            if device.type == "cuda":
                torch.cuda.synchronize()
            fwd_time = time.time() - t0
            totals["fwd"] += fwd_time

            # ----- backward (accumulation) -----
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            loss = loss / accum_iter

            t0 = time.time()
            # NativeScalerWithGradNormCount-style call
            loss_scaler(
                loss,
                optimizer,
                clip_grad=clip_grad,
                parameters=model.parameters(),
                update_grad=update_now,  # all-reduce solo quando non siamo in no_sync
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            totals["bwd"] += (time.time() - t0)

        # ===== 5) zero_grad solo quando aggiorni =====
        if update_now:
            t0 = time.time()
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()
            totals["zero"] += (time.time() - t0)

        # ===== 6) logging =====
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        '''if log_writer is not None and update_now:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)'''

        # ===== 7) tempo iter =====
        totals["iter"] += (time.time() - iter_start)
        iters += 1
        end = time.time()

        # stampa ogni N
        if data_iter_step > 0 and (data_iter_step % log_every_it == 0):
            denom = float(iters)
            print(
                f"[Iter {data_iter_step}] "
                f"data_wait={totals['data']/denom:.3f}s | "
                f"h2d={totals['h2d']/denom:.3f}s | "
                f"gpu_aug={totals['gpu_aug']/denom:.3f}s | "
                f"fwd={totals['fwd']/denom:.3f}s | "
                f"bwd+step={totals['bwd']/ denom:.3f}s | "
                f"zero={totals['zero']/denom:.3f}s | "
                f"iter={totals['iter']/denom:.3f}s"
            )
      

          

    # ===== 8) average =====
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    denom = float(max(1, iters))
    avg_times = {k: (v / denom) for k, v in totals.items()}
    print(
        f"[Epoch {epoch}] AVG — "
        f"data_wait={avg_times['data']:.3f}s | "
        f"h2d={avg_times['h2d']:.3f}s | "
        f"gpu_aug={avg_times['gpu_aug']:.3f}s | "
        f"fwd={avg_times['fwd']:.3f}s | "
        f"bwd+step={avg_times['bwd']:.3f}s | "
        f"zero={avg_times['zero']:.3f}s | "
        f"iter={avg_times['iter']:.3f}s"
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
