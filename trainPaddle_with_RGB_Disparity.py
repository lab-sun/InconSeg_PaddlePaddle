# By Yuxiang Sun, Dec. 4, 2019
# Email: sun.yuxiang@outlook.com

import os, argparse, time, datetime, stat, shutil, sys
import numpy as np


# from torch.utils.data import DataLoader

from util.PD_RGB_Disparity_dataset import PD_RGB_Disparity_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from paddle_model import InconSeg


import paddle
import paddle.optimizer
import paddle.fluid as fluid
import paddle.nn as paddle_F
import paddle.nn.functional as PF
import paddleseg.models.losses as Seglosses
from paddle.optimizer.lr import ExponentialDecay
from paddle.io import DataLoader

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 
parser.add_argument('--model_name', '-m', type=str, default='InconSeg')
parser.add_argument('--batch_size', '-b', type=int, default=2) 
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--need_m', '-need_m', type=int, default=7)
parser.add_argument('--sleep', '-sleep', type=int, default=1)
parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=200) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=3)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
parser.add_argument('--weight_name', '-w', type=str, default='PaddleInconSeg') 
parser.add_argument('--file_name', '-f', type=str, default='pretrain.pdparams')
parser.add_argument('--model_dir', '-wd', type=str, default='./weights_backup/')
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]


def fusion_loss(rgb_seg_1,depth_add,label):

    feature_map_B, feature_map_C, feature_map_W, feature_map_H = rgb_seg_1.shape
    label_B, label_W, label_H = label.shape
    if feature_map_W != label_W:
        label = paddle.cast(label,'float32')
        label = PF.interpolate(label.unsqueeze(1),size=(feature_map_W,feature_map_H),mode="nearest")
        label = paddle.cast(label.squeeze(1),'int64')
    
    criterion1 = Seglosses.CrossEntropyLoss()
    criterion2 = Seglosses.CrossEntropyLoss()
    loss_pr_seg = criterion1(rgb_seg_1, label)
    rgb_seg_1 = rgb_seg_1.detach()
    rgb_seg_1=rgb_seg_1.argmax(1)
    rgb_seg_1 = paddle.equal(rgb_seg_1,label)
    # rgb_seg_1.eq_(label)
    rgb_seg_1=rgb_seg_1.detach()
    add_map = (1-rgb_seg_1.cast('int64'))*label
    add_map=add_map.detach()

    loss_add = criterion2(depth_add,add_map)

    return loss_pr_seg,loss_add,add_map


def train(epo, model, train_dataset, optimizer):
    model.train()

    for it, (images, labels, names) in enumerate(train_loader):

        # if it>10:
        #     break


        images = images.numpy()
        images = paddle.to_tensor(images)

        labels = labels.numpy()
        labels = paddle.to_tensor(labels)



        start_t = time.time() # time.time() returns the current time
        #optimizer.zero_grad()
        depth_result, rgb_result,rgb_seg_f1,depth_add_f1,rgb_seg_f2,depth_add_f2,rgb_seg_f3,depth_add_f3 = model(images)   # depth image result, rgb image result, rgb image pre-segmentaion,  complement feature
        

        criterion1 = Seglosses.CrossEntropyLoss()
        criterion2 = Seglosses.CrossEntropyLoss()
        # labels = labels.unsqueeze(1)

        # depth_result = paddle.transpose(depth_result,[0,2,3,1])
        # labels = paddle.transpose(labels,[0,2,3,1])


        loss_d = criterion1(depth_result, labels)  # Note that the cross_entropy function has already include the softmax function

       
        loss_r = criterion2(rgb_result, labels)  # Note that the cross_entropy function has already include the softmax function

        loss_pr_seg_f1,loss_add_f1,add_map_f1 = fusion_loss(rgb_seg_f1,depth_add_f1,label=labels)
        loss_pr_seg_f2,loss_add_f2,add_map_f2 = fusion_loss(rgb_seg_f2,depth_add_f2,label=labels)
        loss_pr_seg_f3,loss_add_f3,add_map_f3 = fusion_loss(rgb_seg_f3,depth_add_f3,label=labels)
        #print(add_map.size())

        loss = loss_d+loss_r+loss_pr_seg_f1+loss_add_f1+loss_pr_seg_f2+loss_add_f2+loss_pr_seg_f3+loss_add_f3

        loss.backward()
        optimizer.step()

        optimizer.clear_grad()

        lr_this_epo=optimizer.get_lr()
        # for param_group in optimizer.param_groups:
        #     lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, loss_d %.4f, loss_r %.4f, loss_pr_seg_f1 %.4f, loss_add_f1 %.4f, loss_pr_seg_f2 %.4f, loss_add_f2 %.4f,loss_pr_seg_f3 %.4f, loss_add_f3 %.4f, time %s' \
            % (args.model_name, epo, args.epoch_max, it+1, len(train_loader), lr_this_epo, len(names)/(time.time()-start_t), float(loss), float(loss_d), float(loss_r), float(loss_pr_seg_f1), float(loss_add_f1), float(loss_pr_seg_f2), float(loss_add_f2), float(loss_pr_seg_f3), float(loss_add_f3),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', float(loss), accIter['train'])
            writer.add_scalar('Train/loss_d', float(loss_d), accIter['train'])
            writer.add_scalar('Train/loss_r', float(loss_r), accIter['train'])
            writer.add_scalar('Train/loss_seg_f1', float(loss_pr_seg_f1), accIter['train'])
            writer.add_scalar('Train/loss_mse_f1', float(loss_add_f1), accIter['train'])
            writer.add_scalar('Train/loss_seg_f2', float(loss_pr_seg_f2), accIter['train'])
            writer.add_scalar('Train/loss_mse_f2', float(loss_add_f2), accIter['train'])
            writer.add_scalar('Train/loss_seg_f3', float(loss_pr_seg_f3), accIter['train'])
            writer.add_scalar('Train/loss_mse_f3', float(loss_add_f3), accIter['train'])
        view_figure = True # note that I have not colorized the GT and predictions here
        if accIter['train'] % 10 == 0:
            if view_figure:
                input_rgb_images = images.numpy()[1,:3,:,:] # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])


                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = paddle.concat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = groundtruth_tensor.numpy()[1,:3,:,:]
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])

                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                add_map_tensor_f1 = add_map_f1.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                add_map_tensor_f1 = paddle.concat((add_map_tensor_f1, add_map_tensor_f1, add_map_tensor_f1), 1)  # change to 3-channel for visualization
                add_map_images_f1 = add_map_tensor_f1.numpy()[1,:3,:,:]
                writer.add_image('Train/add_map_images_f1', add_map_images_f1, accIter['train'])

                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                add_map_tensor_f2 = add_map_f2.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                add_map_tensor_f2 = paddle.concat((add_map_tensor_f2, add_map_tensor_f2, add_map_tensor_f2), 1)  # change to 3-channel for visualization
                add_map_images_f2 = add_map_tensor_f2.numpy()[1,:3,:,:]
                writer.add_image('Train/add_map_images_f2', add_map_images_f2, accIter['train'])

                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                add_map_tensor_f3 = add_map_f3.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                add_map_tensor_f3 = paddle.concat((add_map_tensor_f3, add_map_tensor_f3, add_map_tensor_f3), 1)  # change to 3-channel for visualization
                add_map_images_f3 = add_map_tensor_f3.numpy()[1,:3,:,:]
                writer.add_image('Train/add_map_images_f3', add_map_images_f3, accIter['train'])

                predicted_depth = depth_result.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_depth = paddle.concat((predicted_depth, predicted_depth, predicted_depth),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_depth_images = predicted_depth.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_depth_images', predicted_depth_images, accIter['train'])

                predicted_rgb = rgb_result.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_rgb = paddle.concat((predicted_rgb, predicted_rgb, predicted_rgb),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_rgb_images = predicted_rgb.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_rgb_images', predicted_rgb_images, accIter['train'])

                predicted_rgb_seg_f1= rgb_seg_f1.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_rgb_seg_f1 = paddle.concat((predicted_rgb_seg_f1, predicted_rgb_seg_f1, predicted_rgb_seg_f1),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_rgb_seg_images_f1 = predicted_rgb_seg_f1.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_rgb_seg_images_f1', predicted_rgb_seg_images_f1, accIter['train'])

                predicted_depth_add_f1= depth_add_f1.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_depth_add_f1 = paddle.concat((predicted_depth_add_f1, predicted_depth_add_f1, predicted_depth_add_f1),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_depth_add_images_f1 = predicted_depth_add_f1.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_depth_add_images_f1', predicted_depth_add_images_f1, accIter['train'])

                predicted_rgb_seg_f2= rgb_seg_f2.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_rgb_seg_f2 = paddle.concat((predicted_rgb_seg_f2, predicted_rgb_seg_f2, predicted_rgb_seg_f2),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_rgb_seg_images_f2 = predicted_rgb_seg_f2.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_rgb_seg_images_f2', predicted_rgb_seg_images_f2, accIter['train'])

                predicted_depth_add_f2= depth_add_f2.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_depth_add_f2 = paddle.concat((predicted_depth_add_f2, predicted_depth_add_f2, predicted_depth_add_f2),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_depth_add_images_f2 = predicted_depth_add_f2.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_depth_add_images_f2', predicted_depth_add_images_f2, accIter['train'])

                predicted_rgb_seg_f3= rgb_seg_f3.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_rgb_seg_f3 = paddle.concat((predicted_rgb_seg_f3, predicted_rgb_seg_f3, predicted_rgb_seg_f3),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_rgb_seg_images_f3 = predicted_rgb_seg_f3.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_rgb_seg_images_f3', predicted_rgb_seg_images_f3, accIter['train'])

                predicted_depth_add_f3= depth_add_f3.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_depth_add_f3 = paddle.concat((predicted_depth_add_f3, predicted_depth_add_f3, predicted_depth_add_f3),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_depth_add_images_f3 = predicted_depth_add_f3.numpy()[1,:3,:,:]
                writer.add_image('Train/predicted_depth_add_images_f3', predicted_depth_add_images_f3, accIter['train'])

                depth_tensor = images[:,3:4] # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                depth_tensor = paddle.concat((depth_tensor, depth_tensor, depth_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                depth_tensor = depth_tensor.numpy()[1,:3,:,:]
                writer.add_image('Train/depth_tensor', depth_tensor, accIter['train'])
        accIter['train'] = accIter['train'] + 1
def validation(epo, model, val_loader): 
    model.eval()
    with paddle.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = images.numpy()
            images = paddle.to_tensor(images)

            labels = labels.numpy()
            labels = paddle.to_tensor(labels)

            start_t = time.time() # time.time() returns the current time
            
            depth_result, rgb_result,rgb_seg_f1,depth_add_f1,rgb_seg_f2,depth_add_f2,rgb_seg_f3,depth_add_f3 = model(images)   # depth image result, rgb image result, rgb image pre-segmentaion,  complement feature
            

            criterion1 = Seglosses.CrossEntropyLoss()
            criterion2 = Seglosses.CrossEntropyLoss()            

            loss_d = criterion1(depth_result, labels)  # Note that the cross_entropy function has already include the softmax function
            loss_r = criterion2(rgb_result, labels)  # Note that the cross_entropy function has already include the softmax function

            loss_pr_seg_f1,loss_add_f1,add_map_f1 = fusion_loss(rgb_seg_f1,depth_add_f1,label=labels)
            loss_pr_seg_f2,loss_add_f2,add_map_f2 = fusion_loss(rgb_seg_f2,depth_add_f2,label=labels)
            loss_pr_seg_f3,loss_add_f3,add_map_f3 = fusion_loss(rgb_seg_f3,depth_add_f3,label=labels)
            #print(add_map.size())

            loss = loss_d+loss_r+loss_pr_seg_f1+loss_add_f1+loss_pr_seg_f2+loss_add_f2+loss_pr_seg_f3+loss_add_f3

            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, loss_d %.4f, loss_r %.4f, loss_pr_seg_f1 %.4f, loss_add_f1 %.4f, loss_pr_seg_f2 %.4f, loss_add_f2 %.4f, loss_pr_seg_f3 %.4f, loss_add_f3 %.4f, time %s' \
                  % (args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss), float(loss_d), float(loss_r), float(loss_pr_seg_f1), float(loss_add_f1), float(loss_pr_seg_f2), float(loss_add_f2), float(loss_pr_seg_f3), float(loss_add_f3),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', float(loss), accIter['val'])
                writer.add_scalar('Validation/loss_d', float(loss_d), accIter['val'])
                writer.add_scalar('Validation/loss_r', float(loss_r), accIter['val'])
                writer.add_scalar('Validation/loss_seg_f1', float(loss_pr_seg_f1), accIter['train'])
                writer.add_scalar('Validation/loss_mse_f1', float(loss_add_f1), accIter['train'])
                writer.add_scalar('Validation/loss_seg_f2', float(loss_pr_seg_f2), accIter['train'])
                writer.add_scalar('Validation/loss_mse_f2', float(loss_add_f2), accIter['train'])
                writer.add_scalar('Validation/loss_seg_f3', float(loss_pr_seg_f3), accIter['train'])
                writer.add_scalar('Validation/loss_mse_f3', float(loss_add_f3), accIter['train'])

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "pothole", "car"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with paddle.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):

            images = images.numpy()
            images = paddle.to_tensor(images)

            labels = labels.numpy()
            labels = paddle.to_tensor(labels)

            depth_result, rgb_result,rgb_seg_f1,depth_add_f1,rgb_seg_f2,depth_add_f2,rgb_seg_f3,depth_add_f3 = model(images)   # depth image result, rgb image result, rgb image pre-segmentaion,  complement feature
            logits = rgb_result 
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.model_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU,F1 = compute_results(conf_total)
    writer.add_scalar('Test/average_precision',precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    writer.add_scalar('Test/average_F1', F1.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s"% label_list[i], recall[i],epo)
        writer.add_scalar('Test(class)/Iou_%s'% label_list[i], IoU[i], epo)
        writer.add_scalar('Test(class)/F1_%s'% label_list[i], F1[i], epo)
    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, pothole, car, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f,%0.4f,%0.4f,%0.4f, ' % (100*precision[i], 100*recall[i], 100*IoU[i], 100*F1[i]))
        f.write('%0.4f,%0.4f,%0.4f,%0.4f\n' % (100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU)),100*np.mean(np.nan_to_num(F1))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

if __name__ == '__main__':
   

    paddle.device.set_device('gpu:0')


    model = eval(args.model_name)(n_class=args.n_class)


    model_dir = os.path.join(args.model_dir, args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.') 
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.') 
    paddle_weight = paddle.load(model_file)
    model.set_state_dict(paddle_weight)


    lr_scheduler = ExponentialDecay(args.lr_start,gamma=args.lr_decay,last_epoch=-1)
    optimizer = paddle.optimizer.SGD(learning_rate=lr_scheduler,parameters=model.parameters())


    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    # preparing folders
    if os.path.exists("./PD_RGB_Disparity"):
        shutil.rmtree("./PD_RGB_Disparity")
    weight_dir = os.path.join("./PD_RGB_Disparity", args.model_name)
    os.makedirs(weight_dir)
    os.chmod(weight_dir, stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
 
    writer = SummaryWriter("./PD_RGB_Disparity/tensorboard_log")
    os.chmod("./PD_RGB_Disparity/tensorboard_log", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./PD_RGB_Disparity", stat.S_IRWXO) 

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = PD_RGB_Disparity_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset  = PD_RGB_Disparity_dataset(data_dir=args.data_dir, split='validation')
    test_dataset = PD_RGB_Disparity_dataset(data_dir=args.data_dir, split='test')

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        #pin_memory  = True,
        drop_last   = True
    )

    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        #pin_memory  = True,
        drop_last   = True
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        #pin_memory   = True,
        drop_last    = True
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))

        train(epo, model, train_dataset, optimizer)
        #validation(epo, model, val_loader)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pdparams')
        print('saving check point %s: ' % checkpoint_model_file)
        paddle.save(model.state_dict(), checkpoint_model_file)

        testing(epo, model, test_loader) # testing is just for your reference, you can comment this line during training

        lr_scheduler.step()
