# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion
# Code for the training of various models (e.g., VQ-VAE, MOF-Constructor, Unconditional & Conditional MOFFUSION)

import os
import time
import inspect

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

from options.train_options import TrainOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import torch
from utils.visualizer import Visualizer

def train_main_worker(opt, model, train_dl, test_dl, test_dl_for_eval, visualizer, device):

    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    train_dg = get_data_generator(train_dl)
    test_dg = get_data_generator(test_dl)

    pbar = tqdm(total=opt.total_iters)

    iter_start_time = time.time()
    for iter_i in range(opt.total_iters):

        opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if get_rank() == 0:
            visualizer.reset()
        
        data = next(train_dg)
        model.set_input(data)
        model.optimize_parameters(iter_i)

        # nBatches_has_trained += opt.batch_size

        if get_rank() == 0:
            if iter_i % opt.print_freq == 0:
                errors = model.get_current_errors()

                t = (time.time() - iter_start_time) / opt.batch_size
                epoch_steps = iter_i
                visualizer.print_current_errors(iter_i, errors, t)

            # if ((nBatches_has_trained % opt.display_freq == 0) or idx == 0):
            # if (nBatches_has_trained % opt.display_freq == 0):

            
            # display every n batches
            if iter_i % opt.display_freq == 0:
                if iter_i == 0 and opt.debug == "1":
                    pbar.update(1)
                    continue

                # eval
                model.inference(data)
                visualizer.display_current_results(model.get_current_visuals(), iter_i, phase='train')

                # test
                test_data = next(test_dg)
                model.inference(test_data)
                visualizer.display_current_results(model.get_current_visuals(), iter_i, phase='test')
 

            if iter_ip1 % opt.save_latest_freq == 0:
                cprint('saving the latest model (current_iter %d)' % (iter_i), 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)

            if iter_ip1 % opt.save_steps_freq == 0:
                cprint('saving the model at iters %d' % iter_ip1, 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)
                cur_name = f'steps-{iter_ip1}'
                model.save(cur_name, iter_ip1)

            if iter_ip1 % opt.save_steps_freq == 0:
                metrics = model.eval_metrics(test_dl_for_eval, global_step=iter_ip1)
                visualizer.print_current_metrics(iter_ip1, metrics, phase='test')
                
                cprint(f'[*] End of steps %d \t Time Taken: %d sec \n%s' %
                    (
                        iter_ip1,
                        time.time() - iter_start_time,
                        os.path.abspath( os.path.join(opt.logs_dir, opt.name) )
                    ), 'blue', attrs=['bold']
                    )

        model.update_learning_rate(model.loss)

        pbar.update(1)
        

if __name__ == "__main__":
    # this will parse args, setup log_dirs, multi-gpus
    opt = TrainOptions().parse_and_setup()

    device = opt.device
    rank = opt.rank

    # CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"]) 
    # import pdb; pdb.set_trace()

    # get current time, print at terminal. easier to track exp
    from datetime import datetime
    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

    train_dl, test_dl, test_dl_for_eval = CreateDataLoader(opt)
    train_ds, test_ds = train_dl.dataset, test_dl.dataset

    dataset_size = len(train_ds)

    cprint('[*] # training mofs = %d' % len(train_ds), 'yellow')
    cprint('[*] # testing mofs = %d' % len(test_ds), 'yellow')

    # main loop

    model = create_model(opt)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    # visualizer
    visualizer = Visualizer(opt)
    if get_rank() == 0:
        visualizer.setup_io()

    # save model and dataset files
    if get_rank() == 0:
        expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
        model_f = inspect.getfile(model.__class__)
        dset_f = inspect.getfile(train_ds.__class__)
        cprint(f'[*] saving model and dataset files: {model_f}, {dset_f}', 'blue')
        modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
        dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
        os.system(f'cp {model_f} {modelf_out}')
        os.system(f'cp {dset_f} {dsetf_out}')

        if opt.vq_cfg is not None:
            vq_cfg = opt.vq_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
            os.system(f'cp {vq_cfg} {cfg_out}')
            
        if opt.df_cfg is not None:
            df_cfg = opt.df_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
            os.system(f'cp {df_cfg} {cfg_out}')

        if opt.res_topo_cfg is not None:
            res_topo_cfg = opt.res_topo_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(res_topo_cfg))
            os.system(f'cp {res_topo_cfg} {cfg_out}')

        if opt.res_BB_cfg is not None:
            res_BB_cfg = opt.res_BB_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(res_BB_cfg))
            os.system(f'cp {res_BB_cfg} {cfg_out}')

    train_main_worker(opt, model, train_dl, test_dl, test_dl_for_eval, visualizer, device)
