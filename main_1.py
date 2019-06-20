# -*- coding: utf-8 -*-

from models import load_model
import my_utils
from torch.utils.data import DataLoader
import torch

args = my_utils.get_args()
data_path,noc = my_utils.get_data_path(args.dataset,'./config.txt')
model = load_model(args.model,noc).cuda()
train_images = data_path+'/train/images'
train_labels = data_path+'/train/labels'
val_images = data_path+'/validation/images'
val_labels = data_path+'/validation/labels'
test_images = data_path+'/test/images'
test_labels = data_path+'/test/labels'
# DATA LOADERS
train_loader = DataLoader(my_utils.getDataset(train_images,
                                              train_labels,
                                              size = (360,480)),
                          batch_size=args.batch_size,
                          num_workers=args.num_of_workers,
                          shuffle=True)

val_loader = DataLoader(my_utils.getDataset(val_images,
                                            val_labels,
                                            size = (360,480)),
                        batch_size=args.batch_size,
                        num_workers=args.num_of_workers,
                        shuffle=False)
test_loader = DataLoader(my_utils.getDataset(test_images,
                                            test_labels,
                                            size = (360,480)),
                        batch_size=args.batch_size,
                        num_workers=args.num_of_workers,
                        shuffle=False)

# TRAINING
epochs = args.max_epochs
save_path = args.save_path

train_flag = int(args.fresh_train)

if train_flag:
    trainer = my_utils.Trainer(model,train_loader,val_loader,save_path,epochs,noc)
    trained_model = trainer.train()
else:
    print('Loading model')
    trained_model = torch.load(save_path+'/best_model.pth')
    print('model loaded')
tester_train = my_utils.Tester(trained_model,train_loader,save_path+'/eval_train',noc)
tester_train.test()
tester_val = my_utils.Tester(trained_model,val_loader,save_path+'/eval_val',noc)
tester_val.test()
tester_test = my_utils.Tester(trained_model,test_loader,save_path+'/eval_test',noc)
tester_test.test()