import torch
import datetime
import time
from clustering.augs import IMAGE_AUGMENTATIONS, EMBED_AUGMENTATIONS, AugWrapper
import clustering.utils as utils
import torch.backends.cudnn as cudnn
from clustering import multihead_losses as losses
from collections import OrderedDict
import torch.nn as nn
import inspect
from clustering.multi_head import MultiHeadClassifier
import numpy as np
import random
from torch.utils.data import Dataset
import math
import sys
from types import SimpleNamespace
from typing import Union, Tuple

class EmbedNN(Dataset):
    def __init__(self,
                 features,
                 labels=None,
                 knn=None,
                 transform=None,
                 k=10):
        """
        Args:
            features (Tensor): [N, D] precomputed embeddings
            labels (Tensor or None): [N] class labels, or None to auto-generate dummy labels
            knn (Tensor): [N, K] precomputed neighbor indices
            transform (callable): A function taking (anchor, neighbor) or (anchor, *neighbors)
            k (int): Number of neighbors to sample from
        """
        super().__init__()
        self.features = features
        self.labels = labels if labels is not None else torch.zeros(len(features), dtype=torch.long)
        self.knn = knn
        self.transform = transform
        self.k = k
        if self.knn is None:
            raise ValueError("knn must be provided")
        if k < 0:
            self.k = self.knn.size(1)
        else:
            self.k = min(k, self.knn.size(1))
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        neighbor_indices = self.knn[idx][:self.k]
        neighbor_idx = random.choice(neighbor_indices.tolist())
        anchor = self.features[idx]
        neighbor = self.features[neighbor_idx]
        label = self.labels[idx]
        if self.transform:
            return self.transform(anchor, neighbor), label
        else:
            return (anchor, neighbor), label

class TeacherStudentCombo(nn.Module):
    def __init__(self, student, teacher, args):
        super().__init__()
        # synchronize batch norms (if any)
        if utils.has_batchnorms(student) and not args.disable_ddp:
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # teacher and student start with the same weights
        teacher.load_state_dict(student.state_dict())
        # Hacky
        if not args.train_backbone:
            student.backbone = teacher.backbone
        elif not args.req_grad:
            print('WARNING: args.train_backbone=True, but args.req_grad=False. '
                  'This is probably not what you want.')
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        if getattr(args, "verbose", False):
            print(f"Student and Teacher are built: they are both {args.arch} network.")
        self.args = args
        self.student = student
        self.teacher = teacher
    def forward(self, images):
        if self.args.train_backbone:
            return self.teacher(images), self.student(images)
        embed = self.teacher.backbone_embed(images)
        return self.teacher.apply_head(embed), self.student.apply_head(embed)
    @property
    def module(self):
        return self
    def student_dict(self):
        if self.args.train_backbone:
            return self.student.state_dict()
        return OrderedDict([(k, v) for k, v in self.student.state_dict().items() if "backbone" not in k])
    @property
    def trainable_student(self):
        if self.args.train_backbone:
            return self.student
        return self.student.head
    def teacher_dict(self):
        if self.args.train_backbone:
            return self.teacher.state_dict()
        return OrderedDict([(k, v) for k, v in self.teacher.state_dict().items() if "backbone" not in k])
    @property
    def trainable_teacher(self):
        if self.args.train_backbone:
            return self.teacher
        return self.teacher.head


def load_model(config, head=True, split_preprocess=False):
    """
    Minimal load_model for precomputed embeddings.
    Returns a MultiHeadClassifier if head=True, else just the placeholder backbone name.
    """
    if not config.precomputed:
        raise ValueError("This version of load_model only supports precomputed=True")
    backbone = config.arch  # Typically 'custom' or a placeholder
    if getattr(config, "verbose", False):
        print("Using precomputed embeddings from", config.arch)
    if head:
        if getattr(config, "embed_dim", None) is None:
            raise ValueError("embed_dim must be set for head construction")
        mmc_params = inspect.signature(MultiHeadClassifier).parameters
        mmc_args = {k: v for k, v in config.__dict__.items() if k in mmc_params}
        model = MultiHeadClassifier(backbone, **mmc_args)
        if getattr(config, "verbose", False):
            print("Head loaded.")
    else:
        model = backbone
    if split_preprocess:
        return model, None, None
    return model, None




def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_args(out_dim, embed_dim):
    return SimpleNamespace(
        # core training setup
        out_dim=out_dim,
        embed_dim=embed_dim,
        num_heads=16,
        knn=25,
        precomputed=True,
        arch='custom',
        loss='TEMI',
        loss_args={"beta": 0.6},

        # augmentation
        embed_aug='none',
        num_augs=1,
        aug_args={},

        # training behavior
        batch_size_per_gpu=512,
        seed=0,
        new_run=True,
        train_backbone=False,
        disable_ddp=True,
        use_fp16=False,

        # DINO-specific settings
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=30,
        optimizer='adamw',
        weight_decay=0.0001,
        weight_decay_end=0.0005,
        momentum_teacher=0.9995,
        max_momentum_teacher=0.9995,
        clip_grad=0,
        freeze_last_layer=1,

        # scheduler
        epochs=100,
        lr=1e-4,
        min_lr=1e-4,
        warmup_epochs=20,

        # misc
        loader='EmbedNN',
        loader_args={},
        verbose=False,
    )


@torch.no_grad()
def compute_neighbors(embedding, k, verbose=False):
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    num_embeds = embedding.shape[0]
    if num_embeds <= 8 * 1e4:
        dists = embedding @ embedding.permute(1, 0)
        dists.fill_diagonal_(-torch.inf)
        return dists.topk(k, dim=-1)
    else:
        topk_knn_ids = []
        topk_knn_dists = []
        if verbose:
            print("Chunk-wise implementation of k-nn in GPU")
        step_size = 64
        embedding = embedding.cuda()
        for idx in range(0, num_embeds, step_size):
            idx_next_chunk = min((idx + step_size), num_embeds)
            features = embedding[idx:idx_next_chunk, :]
            dists_chunk = torch.mm(features, embedding.T).cpu()
            dists_chunk.fill_diagonal_(-torch.inf)
            max_dists, indices = dists_chunk.topk(k, dim=-1)
            topk_knn_ids.append(indices)
            topk_knn_dists.append(max_dists)
        return torch.cat(topk_knn_dists), torch.cat(topk_knn_ids)


def train_one_epoch(student_teacher_model, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ", enabled=getattr(args, "verbose", False))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images, _ = data
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        
        # teacher and student forward passes + compute dino loss
        with torch.amp.autocast("cuda", enabled=fp16_scaler is not None):
            teacher_out, student_out = student_teacher_model(images)
            if losses.is_multihead(dino_loss) or args.num_heads == 1:
                head_losses = dino_loss(student_out, teacher_out, epoch=epoch)
            else:
                head_losses = torch.stack([d(s, t, epoch=epoch) for d, s, t in zip(dino_loss, student_out, teacher_out)])
            loss = head_losses.mean()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), flush=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student_teacher_model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student_teacher_model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student_teacher_model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student_teacher_model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            s_head_params = student_teacher_model.module.trainable_student.parameters()
            t_head_params = student_teacher_model.module.trainable_teacher.parameters()
            for param_q, param_k in zip(s_head_params, t_head_params):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update_raw(head_losses=head_losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()

    if utils.is_main_process() and args.num_heads > 1:
        avg_loss = metric_logger.meters['head_losses'].global_avg
        student_teacher_model.module.teacher.head.set_losses(avg_loss)
        student_teacher_model.module.student.head.set_losses(avg_loss)

    if utils.is_main_process():
        if args.num_heads == 1:
            pass
        else:
            avg_loss = metric_logger.meters['head_losses'].global_avg
        d_loss = dino_loss[0] if hasattr(dino_loss, "__getitem__") else dino_loss
    if getattr(args, "verbose", False):
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.scalar_meters.items()}


def train_dino(args, features, knn, progress_callback=None):
    
    fix_random_seeds(args.seed)
    cudnn.benchmark = True
    verbose = getattr(args, "verbose", False)
    if verbose:
        print('pre loading')
    student, _, normalize = load_model(args, head=True, split_preprocess=True)
    if verbose:
        print('loading1')
    teacher, _ = load_model(args)
    if verbose:
        print('loading2')
    if not args.precomputed:
        aug = IMAGE_AUGMENTATIONS[args.image_aug](num_augs=args.num_augs, **args.aug_args)
        transform = AugWrapper(
            vit_image_size=args.vit_image_size,
            aug_image_size=args.aug_image_size,
            global_augs=aug,
            normalize=normalize,
            image_size=args.image_size
        )
    else:
        aug = EMBED_AUGMENTATIONS[args.embed_aug](num_augs=args.num_augs, **args.aug_args)
        transform = AugWrapper(global_augs=aug)
    dataset = EmbedNN(features, None, knn, transform=transform, k=args.knn)
    sampler = None
    # if len(dataset) < 10*args.batch_size_per_gpu: 
    #     check_drop = False
    # else:
    #     check_drop = True
    check_drop = len(dataset) % args.batch_size_per_gpu != 0
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=(sampler is None),
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=0,
        pin_memory=True,
        drop_last=check_drop,
    )
    if verbose:
        print(f"In-distribution Data loaded: there are {len(dataset)} images.")
        print("len dataloader", len(data_loader))
    student_teacher_model = TeacherStudentCombo(teacher=teacher, student=student, args=args)
    student_teacher_model = student_teacher_model.cuda()
    loss_class = getattr(losses, args.loss)
    dino_loss_args = dict(
        out_dim=args.out_dim,
        batchsize=args.batch_size_per_gpu,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        num_heads=16,
        **args.loss_args
    )
    if losses.is_multihead(loss_class):
        dino_loss = loss_class(**dino_loss_args).cuda()
    elif args.num_heads == 1:
        dino_loss = loss_class(**dino_loss_args).cuda()
    else:
        dino_loss = torch.nn.ModuleList([loss_class(**dino_loss_args) for _ in range(args.num_heads)]).cuda()
    params_groups = utils.get_params_groups(student_teacher_model.module.trainable_student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    else:
        raise ValueError("Unknown optimizer: {}".format(args.optimizer))
    fp16_scaler = None
    # if args.use_fp16:
    #     fp16_scaler = torch.cuda.amp.GradScaler()
    bs_factor = (args.batch_size_per_gpu * utils.get_world_size()) / 256.
    lr_schedule = utils.cosine_scheduler(
        args.lr * bs_factor,
        args.min_lr * bs_factor,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, args.max_momentum_teacher,
                                               args.epochs, len(data_loader))
    if verbose:
        print("Loss, optimizer and schedulers ready.")
    start_epoch = 0
    start_time = time.time()
    if verbose:
        print("Starting DINO training!")
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            student_teacher_model, dino_loss, data_loader,
            optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args
        )
        if progress_callback:
            progress_callback(epoch + 1, args.epochs)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if verbose:
        print('Training time {}'.format(total_time_str))
    return student_teacher_model.module.student

@torch.no_grad()
def run_eval_pipeline(model, features, threshold, use_head=True):
    model.eval()
    features = features.cuda()

    if use_head:
        try:
            outputs, _ = model(features)
        except Exception:
            outputs = model(features)
    else:
        outputs = features  # skip model if you're already using head outputs

    outputs = outputs.cpu()
    _, max_indices = torch.max(outputs, dim=1)
    max_indices = max_indices.numpy()
    all_indices = outputs.numpy()

    def generate_hypergraph(all_indices, t):
        hypergraph = np.where(all_indices < t, 0, 1)
        for i in range(hypergraph.shape[0]):
            if np.all(hypergraph[i] == 0):
                hypergraph[i, np.argmax(all_indices[i])] = 1
        return hypergraph

    return generate_hypergraph(all_indices, threshold)

def make_knn(features,num_k=25, verbose=False):
    nn_dists, knn = compute_neighbors(features, num_k, verbose=verbose)
    return nn_dists, knn

def train_model(features: Union[np.ndarray, torch.Tensor], out_dim: int, progress_callback=None, verbose=False) -> nn.Module:
    """
    Train the clustering model on the input features.

    Args:
        features: A [N, D] array or tensor of precomputed feature vectors.
        out_dim: The output dimensionality of the model (i.e., number of output nodes).

    Returns:
        A trained model (nn.Module) ready for inference.
    """
    features = torch.tensor(features, dtype=torch.float32) if not torch.is_tensor(features) else features
    args = get_args(out_dim, features.shape[1])
    args.verbose = verbose
    if args.batch_size_per_gpu > len(features):
        args.batch_size_per_gpu = len(features)//2

    _, knn = compute_neighbors(features, args.knn, verbose=args.verbose)
    return train_dino(args, features, knn, progress_callback)

@torch.no_grad()
def generate_hypergraph(
    model: nn.Module,
    features: Union[np.ndarray, torch.Tensor],
    threshold: float
) -> np.ndarray:
    """
    Generate a hypergraph using the trained model's output probabilities.

    Args:
        model: A trained clustering model.
        features: [N, D] embeddings.
        threshold: Float threshold for softmax probabilities.

    Returns:
        Binary hypergraph (np.ndarray of shape [N, out_dim])
    """
    features = torch.tensor(features, dtype=torch.float32).cuda() if not torch.is_tensor(features) else features.cuda()
    model.eval()
    try:
        outputs, _ = model(features)
    except Exception:
        outputs = model(features)
    outputs = outputs.cpu().numpy()

    hypergraph = np.where(outputs < threshold, 0, 1)
    for i in range(hypergraph.shape[0]):
        if np.all(hypergraph[i] == 0):
            hypergraph[i, np.argmax(outputs[i])] = 1
    return hypergraph

# def get_hypergraph(features,out_dim,theshold):
#     # features = torch.load(r"F:\PhD\Projects\HyperGallery\data\embeddings\mh17\feature_vectors.pt").cpu().numpy()
#     features = torch.tensor(features, dtype=torch.float32)
#     args = get_args(out_dim)
#     nn_dists, knn = make_knn(features,num_k=25) 
#     model = train_dino(args, features, knn)
#     hypergraph = run_eval_pipeline(model, features, threshold=0.5)
#     return hypergraph

def temi_cluster(
    features: Union[np.ndarray, torch.Tensor],
    out_dim: int,
    threshold: float,
    *,
    progress_callback=None,
    verbose=False,
) -> Tuple[np.ndarray, nn.Module]:
    """
    Convenience wrapper to train model and generate hypergraph in one call.

    Returns:
        - hypergraph (np.ndarray): Clustering result
        - model (nn.Module): Trained model
    """
    model = train_model(features, out_dim, progress_callback, verbose=verbose)
    hypergraph = generate_hypergraph(model, features, threshold)
    return hypergraph, model

if __name__ == "__main__":
    dummy = np.random.randn(500, 1536).astype(np.float32)
    hypergraph, model = temi_cluster(dummy, out_dim=10, threshold=0.5, verbose=True)
    print("Hypergraph shape:", hypergraph.shape)


__all__ = ["temi_cluster", "train_model", "generate_hypergraph"]