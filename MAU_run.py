import os
import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer
# import pynvml


# pynvml.nvmlInit()
# -----------------------------------------------------------------------------

from configs.radar_train_configs import configs

parser = configs()
args = parser.parse_args()
args.tied = True





def train_wrapper(model):
    begin = 0
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])

    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_val_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)  # 训练到5000次的时候就不适用答案增强了
    itr = begin
    # real_input_flag = {}
    for epoch in range(0, args.max_epoches):
        if itr > args.max_iterations:
            break
        for ims in train_input_handle:
            if itr > args.max_iterations:
                break
            # batch_size = ims.shape[0]
            # eta, real_input_flag = schedule_sampling(eta, itr, args.img_channel, batch_size)

            if itr == 0:
                print('Validate:')
                trainer.test(model, val_input_handle, args, itr)

            trainer.train(model, ims, None, args, itr)
            if itr % args.snapshot_interval == 0 and itr > begin:
                model.save(itr)

            if itr % args.test_interval == 0 and itr != 0:
                print('Validate:')
                trainer.test(model, val_input_handle, args, itr)

            itr += 1

            # meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("GPU memory:%dM" % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)))


def text_Wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)

    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr)


if __name__ == '__main__':

    print('Initializing models')
    if args.is_training == 'True':
        args.is_training = True
    else:
        args.is_training = False

    model = Model(args)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.network.parameters()) / 1000000.0))

    if args.is_training:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        train_wrapper(model)
    else:
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        text_Wrapper(model)
