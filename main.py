import copy
import json
import os
import warnings
from absl import app, flags
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, GrowingGaussianDiffusionTrainer, GrowingGaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

from torchviz import make_dot


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('get_ckpt', False, help='Retrieve checkpoint to continue training')
# Finetuning
flags.DEFINE_bool('finetune', False, help='fine tuning from pretrained')

# Progressive Diffusion
flags.DEFINE_bool('progressive_train',False,help='progressive training')
flags.DEFINE_bool('progressive_eval',False, help='progressive evaluation')

# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
# Utilitaries 
flags.DEFINE_bool('model_graph', False, help='generate and save a graph of the model')


device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def ema_list(source_list, target_list, decay):
    if not(len(source_list)==len(target_list)):
        return
    for source, target in zip(source_list,target_list):
        ema(source,target,decay)

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            if step==0 and FLAGS.model_graph:
                make_dot(loss,dict(trainer.model.named_parameters())).render('graph',format='png')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)



def finetune():
    # dataset TODO 
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    client_1_idx = np.where((np.array(dataset.targets) == 0))[0]
    client_1_ds = torch.utils.data.Subset(dataset, client_1_idx)

    dataloader = torch.utils.data.DataLoader(
        client_1_ds, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    net_model.load_state_dict(ckpt['net_model'])

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            if step==0 and FLAGS.model_graph:
                make_dot(loss,dict(trainer.model.named_parameters())).render('graph',format='png')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()

def progressive_train():
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    ch_mult=FLAGS.ch_mult
    ch=FLAGS.ch
    T=FLAGS.T
    attn=FLAGS.attn
    num_res_blocks=FLAGS.num_res_blocks
    dropout=FLAGS.dropout
    model2 = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    ch_mult.pop()
    model1 = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    ch_mult.pop()
    model0 = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    
    model_list=[model0,model1,model2]
    model_list=[model.to(device) for model in model_list]
    ema_model_list=[copy.deepcopy(model) for model in model_list]

    optim_list =[torch.optim.Adam(model.parameters(), lr=FLAGS.lr) for model in model_list]

    sched_list = [torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr) for optim in optim_list]


    trainer = GrowingGaussianDiffusionTrainer(
        model_list, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    sampler = GrowingGaussianDiffusionSampler(
        model_list, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GrowingGaussianDiffusionSampler(
        ema_model_list, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    #if FLAGS.parallel:
    #    trainer = torch.nn.DataParallel(trainer)
    #    net_sampler = torch.nn.DataParallel(net_sampler)
    #   ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, int(FLAGS.img_size/4), int(FLAGS.img_size/4))
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # show model size
    model_size = 0
    for model in model_list:
        for param in model.parameters():
            model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            for optim in optim_list:
                optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()

            for model in model_list:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), FLAGS.grad_clip)
            for optim in optim_list:
                optim.step()
            for sched in sched_list:
                sched.step()
            ema_list(model_list, ema_model_list, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                for model in model_list:
                    model.eval()
                with torch.no_grad():
                    x_0 = sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                for model in model_list:
                    model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'model_list': [model.state_dict() for model in model_list],
                    'ema_model_list': [ema_model.state_dict() for ema_model in ema_model_list],
                    'sched': [sched.state_dict() for sched in sched_list],
                    'optim': [optim.state_dict() for optim in optim_list],
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = progressive_evaluate(net_sampler, model_list)
                ema_IS, ema_FID, _ = progressive_evaluate(ema_sampler, ema_model_list)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()

def progressive_evaluate(sampler,model_list):
    for model in model_list:
        model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, int(FLAGS.img_size/4), int(FLAGS.img_size/4)))
            batch_images = sampler(x_T.to(device)).cpu()
            print(batch_images.shape)
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def progressive_eval():
    # model setup
    ch_mult=FLAGS.ch_mult
    ch=FLAGS.ch
    T=FLAGS.T
    attn=FLAGS.attn
    num_res_blocks=FLAGS.num_res_blocks
    dropout=FLAGS.dropout
    model2 = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    ch_mult.pop()
    model1 = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    ch_mult.pop()
    model0 = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)


    model_list=[model0,model1,model2]
    model_list=[model.to(device) for model in model_list]

    ckpt_list = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))['model_list']
    for i in range(3):
        model_list[i].load_state_dict(ckpt_list[i])

    sampler = GrowingGaussianDiffusionSampler(
        model_list, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    #if FLAGS.parallel:
    #    sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate

    (IS, IS_std), FID, samples = progressive_evaluate(sampler, model_list)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    #model.load_state_dict(ckpt['ema_model'])
    #(IS, IS_std), FID, samples = progressive_evaluate(sampler, model)
    #print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    #save_image(
    #    torch.tensor(samples[:256]),
    #    os.path.join(FLAGS.logdir, 'samples_ema.png'),
    #    nrow=16)
    


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    elif FLAGS.eval:
        eval()
    elif FLAGS.finetune:
        finetune()
    elif FLAGS.progressive_train:
        progressive_train()
    elif FLAGS.progressive_eval:
        progressive_eval()
    else:
        print('Add --train and/or --eval and/or --finetune and/or a progressive variant of these to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
