import pdb
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import Dataset
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast

from typing import Any, Tuple, Dict, Optional, Union
from torch.nn.functional import one_hot, softmax
from functools import partial
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from enum import Enum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


# --------------------------------- KNN Eval ---------------------------------


class SingleOutputModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        if self.model.output_dict:
            return self.model(samples)["image_features"]
        else:
            return self.model(samples)[0]


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value
    

def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


class DatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def get_image_data(self, index: int) -> bytes:
        return self._dataset.get_image_data(index)

    def get_target(self, index: int) -> Tuple[Any, int]:
        target = self._dataset.get_target(index)
        return (index, target)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        # try:
        image, target = self._dataset[index]
        target = index if target is None else target
        return image, (index, target)
        # except:
        #     return None, (None, None)

    def __len__(self) -> int:
        return len(self._dataset)
    

class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()

        self.device = device
        self.train_features_rank_T = train_features.T.to(self.device)
        self.candidates = train_labels.view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        # torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        # if self.global_rank != source_rank:
        # broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        # torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        # if self.global_rank == target_rank:
        topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(1)]
        retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(1)]
        # pdb.set_trace()
        # torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        # torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)
        for i in range(len(topk_sims_rank)):
            topk_sims_rank[i] = topk_sims
            retrieved_rank[i] = neighbors_labels

        # if self.global_rank == target_rank:
        # Perform a second top-k on the k * global_size retrieved neighbors
        topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
        retrieved_rank = torch.cat(retrieved_rank, dim=1)
        results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
        return results
        # return None

    def compute_neighbors(self, features_rank):
        for rank in range(1):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k
    

@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False, loc=None):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    features, all_labels = None, None
    for samples, (index, labels_rank) in tqdm(data_loader, desc="Extracting features"):
        samples = samples.to('cuda')
        labels_rank = labels_rank.to('cuda')
        index = index.to('cuda')
        # if model.output_dict:
        #     features_rank = model(samples)['image_features'].float()
        # else:
        #     features_rank = model(samples)[0].float()
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            print(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = index.to(gather_device)
        features_all_ranks = features_rank.to(gather_device)
        labels_all_ranks = labels_rank.to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)
            
    print(f"Features shape: {tuple(features.shape)}")
    print(f"Labels shape: {tuple(all_labels.shape)}")
    
    assert torch.all(all_labels > -1)
    
    if loc is not None:
        torch.save({'train_features': features, 'train_labels': all_labels}, loc)

    return features, all_labels


def extract_features(model, dataloader, gather_on_cpu=False, loc=None):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataloader.dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = torch.utils.data.DataLoader(
            dataset_with_enumerated_targets,
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers
        )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu, loc=loc)


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping

class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


@torch.inference_mode()
def knn_evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    # metric_logger = MetricLogger(delimiter="  ")
    # header = "Test:"

    for samples, targets, *_ in tqdm(data_loader, desc='evaluating the Dataset with Knn'):
        # pdb.set_trace()
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            # metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    # metric_logger.synchronize_between_processes()
    # print(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    # metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # return metric_logger_stats, stats
    return stats


def zeroshot_eval_knn_model(
        model, train_loader, eval_loader, device=None, 
        gather_on_cpu=False, loc=None, use_existing_encs=True,
        nb_knn=(10, 20, 100, 200)
    ):
    """Evaluate a model with Knn on a dataset"""
    if loc is not None and os.path.exists(loc) and use_existing_encs:
        _ = torch.load(loc)
        train_features, train_labels = _['train_features'], _['train_labels']
    else:
        train_features, train_labels = extract_features(model, train_loader, gather_on_cpu=gather_on_cpu, loc=loc)
    # train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
    print("Train features are shape: ", train_features.shape)
    print("Train labels are shape: ", train_labels.shape)
    # logging.info("Train features are shape: ", train_features.shape)
    # logging.info("Train labels are shape: ", train_labels.shape)
    # pdb.set_trace()
    num_classes = train_labels.max() + 1
    metric_collection = build_topk_accuracy_metric(AccuracyAveraging.MEAN_ACCURACY, num_classes=num_classes)
    device = torch.cuda.current_device()
    partial_module = partial(KnnModule, T=.07, device=device, num_classes=num_classes)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=[-1],
        n_tries=1,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    # pdb.set_trace()
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    # pdb.set_trace()
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)
    # print("Start the k-NN classification.")
    logging.info("Start the k-NN classification.")
    results_dict = knn_evaluate(model_with_knn, eval_loader, postprocessors, metrics, device)
    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]
    results_dict_knn = {}
    for knn_ in results_dict.keys():
        top1 = results_dict[knn_]["top-1"].item() * 100.0
        top5 = results_dict[knn_]["top-5"].item() * 100.0
        results_dict_knn[f"{knn_} Top 1"] = top1
        results_dict_knn[f"{knn_} Top 5"] = top5
        # print(f"{knn_} classifier result: Top1: {top1:.2f} Top5: {top5:.2f}")
        logging.info(f"{knn_} classifier result: Top1: {top1:.2f} Top5: {top5:.2f}")
    
    return_dict = {}
    for (_, nn) in results_dict.keys():
        for key, val in results_dict[(_, nn)].items():
            if key not in return_dict:
                new_key = f"NN-{nn}-{key}"
                return_dict[new_key] = val.item()

    return return_dict


def build_dataloaders(data_config, device='cuda'):
    """ Load all dataloaders required for experiment. """
    if isinstance(data_config, list):
        return [build_dataloaders(c, device) for c in data_config]
    
    data_config['device'] = device
    
    if data_config['name'] == 'eurosat':
        from open_clip_train.eval_datasets.eurosat import prepare_train_loaders, prepare_test_loaders
    elif data_config['name'] == 'dtd':
        from open_clip_train.eval_datasets.dtd import prepare_train_loaders, prepare_test_loaders
    elif data_config['name'] == 'sun397':
        from open_clip_train.eval_datasets.sun397 import prepare_train_loaders, prepare_test_loaders
    elif data_config['name'] == 'resisc45':
        from open_clip_train.eval_datasets.resisc45 import prepare_train_loaders, prepare_test_loaders
    elif data_config['name'] == 'pets':
        from open_clip_train.eval_datasets.pets import prepare_train_loaders, prepare_test_loaders
    elif data_config['name'] == 'flowers':
        from open_clip_train.eval_datasets.flowers import prepare_train_loaders, prepare_test_loaders
    elif data_config['name'] == 'food':
        from open_clip_train.eval_datasets.food import prepare_train_loaders, prepare_test_loaders
    elif data_config['name'] == 'cars':
        from open_clip_train.eval_datasets.cars import prepare_train_loaders, prepare_test_loaders
    # elif data_config['name'] == 'mnist':
    #     from eval_datasets.mnist import prepare_train_loaders, prepare_test_loaders
    # elif data_config['name'] == 'gtsrb':
    #     from eval_datasets.gtsrb import prepare_train_loaders, prepare_test_loaders
    # elif data_config['name'] == 'svhn':
    #     from eval_datasets.svhn import prepare_train_loaders, prepare_test_loaders
    # elif data_config['name'] == 'imagenet':
    #     from eval_datasets.imagenet import prepare_train_loaders, prepare_test_loaders
    # elif data_config['name'] == 'cc3m':
    #     from eval_datasets.cc3m import prepare_train_loaders, prepare_test_loaders
    else:
        raise NotImplementedError(data_config['name'])
    
    train_loaders = prepare_train_loaders(data_config)
    test_loaders = prepare_test_loaders(data_config)
    try:
        return {
            'name': data_config['name'],
            'train': train_loaders,
            'test': test_loaders
        }
    except:
        import pdb; pdb.set_trace()


def run_knn_evals(model, args, tb_writer, preprocessor, epoch, step):
    knn_model = SingleOutputModel(model)
    datasets_to_eval = ['eurosat', 'cars', 'resisc45', 'pets', 'food', 'flowers', 'dtd']
    all_dataloaders = build_dataloaders(
        [
            {
                "name": name,
                "batch_size": 128,
                "num_workers": 4,
                "train_preprocess": preprocessor,
                "eval_preprocess": preprocessor
            } for name in datasets_to_eval
        ]
    )
    all_metrics = {"epoch": epoch}
    for name, train_test_loaders in zip(datasets_to_eval, all_dataloaders):
        train_loader = train_test_loaders['train']['full']
        test_loader = train_test_loaders['test']['test']

        knn_results = zeroshot_eval_knn_model(knn_model, train_loader, test_loader)

        for key, val in knn_results.items():
            all_metrics[f"knn_val/{name}/{key}"] = val

    if args.save_logs:
        if tb_writer is not None:
            for name, val in all_metrics.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "knn_results.jsonl"), "a+") as f:
            f.write(json.dumps(all_metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        # log_data['epoch'] = epoch
        wandb.log(all_metrics, step=step)