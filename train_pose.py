import argparse
from training.train_common import prepare, train, validate, save_network_input_output, test_augmentation_speed
from training.ds_generators import DataGeneratorClient, DataIterator
from config import COCOSourceConfig, GetConfig

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, default='train', help='task')
parser.add_argument('--config_name', type=str, default='Canonical', help='config name')
parser.add_argument('--experiment_name', type=str, default=None, help='experiment name')
parser.add_argument('--epoch', type=int, default=0, help='epoch')
parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
parser.add_argument('--datapath', type=str, default='dataset', help='path to the dataset')
parser.add_argument('--modelpath', type=str, default='/mnt/models', help='path to the dataset')
parser.add_argument('--logpath', type=str, default='/mnt/logs', help='path to the dataset')
parser.add_argument('--use_client_gen', action='store_true', help='use rmpe server')
args = parser.parse_args()


task = args.task
config_name = args.config_name
experiment_name = args.experiment_name
epoch = args.epoch
datapath = args.datapath
modelpath = args.modelpath
logpath = args.logpath
batch_size = args.batch_size
use_client_gen = False

config = GetConfig(config_name)

train_client = DataIterator(config, COCOSourceConfig(datapath + "/coco_train_dataset.h5"), shuffle=True,
                            augment=True, batch_size=batch_size)
val_client = DataIterator(config, COCOSourceConfig(datapath + "/coco_val_dataset.h5"), shuffle=False, augment=False,
                          batch_size=batch_size)

train_samples = train_client.num_samples()
val_samples = val_client.num_samples()

model, iterations_per_epoch, validation_steps, epoch, metrics_id, callbacks_list = \
    prepare(config=config, config_name=config_name, exp_id=experiment_name, train_samples = train_samples, val_samples = val_samples, batch_size=batch_size, epoch=epoch, logpath=logpath, modelpath=modelpath)


if task == "train":
    train(config, model, train_client, val_client, iterations_per_epoch, validation_steps, metrics_id, epoch, use_client_gen, callbacks_list)

elif task == "validate":
    validate(config, model, val_client, validation_steps, metrics_id, epoch)

elif task == "save_network_input_output":
    save_network_input_output(model, val_client, validation_steps, metrics_id, batch_size, epoch)

elif task == "save_network_input":
    save_network_input_output(None, val_client, validation_steps, metrics_id, batch_size)

elif task == "test_augmentation_speed":
    test_augmentation_speed(train_client)
