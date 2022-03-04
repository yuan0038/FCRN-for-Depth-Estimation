
from loss_fn import *

from metrics import Result, AverageMeter
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.utils.data import DataLoader
torch.manual_seed(40)
import torch.utils.data
from models.model import FCRN



fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3']

best_result = Result()
best_result.set_to_worst()



def create_dataloaders(args):
    print("=> creating Data loaders ...")
    traindir = 'Data/nyudepthv2/train'
    valdir = 'Data/nyudepthv2/val'

    train_loader = None
    val_loader = None

    from Dataset.myDataset import NYUDataset
    if not args.evaluate:
        train_dataset = NYUDataset(traindir,split='train')
        train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.workers,
                                  shuffle=True,pin_memory=True,drop_last=True)

    val_dataset = NYUDataset(valdir, split='val')
    # set batch size to be 1 for validation
    val_loader = DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=args.workers
                            , pin_memory=True)
    print("=> Data loaders created.")
    return train_loader, val_loader


def train(train_loader,model,loss_fn,optimizer,epoch):
    average_meter = AverageMeter()
    model.train()

    train_bar=Progrecess_bar(len(train_loader),'train progress')
    count = 0
    running_loss = 0
    for i, (input, target) in enumerate(train_loader):

        input, target = input.cuda(), target.cuda()

        # compute pred
        pred = model(input)
        loss = loss_fn(pred, target)
        count += 1

        running_loss += loss.data.cpu().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, input.size(0))
        train_bar(count)


    epoch_loss = running_loss / count
    avg=average_meter.average()
    print('\tEpoch {} train loss:{:.5f}'.format(epoch,epoch_loss))
    print('RMSE\tMAE\tDelta1\tDelta2\tDelta3\tREL\tLg10\t')
    print('{average.rmse:.3f}\t'
          '{average.mae:.3f}\t'
          '{average.delta1:.3f}\t'
          '{average.delta2:.3f}\t'
          '{average.delta3:.3f}\t'
          '{average.absrel:.3f}\t'
          '{average.lg10:.3f}\t'
          .format(average=avg))
    
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                         'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3})


def validate(val_loader,model,loss_fn,epoch,write_to_file=True):

    average_meter = AverageMeter()
    model.eval()

    val_bar = Progrecess_bar(len(val_loader), 'val   progress')

    if args.evaluate:
        val_imgs_metric_saver = Val_imgs_metrics_Saver(output_directory, fieldnames)
        saved_img_path=os.path.join(output_directory,'saved_imgs')
        if not os.path.exists(saved_img_path):
            os.makedirs(saved_img_path)
    running_loss = 0
    count=0
    with torch.no_grad():
        for  (input, target) in val_loader:
            input, target = input.cuda(), target.cuda()

            # compute output
            pred = model(input)
            loss = loss_fn(pred, target)
            running_loss += loss.data.cpu().numpy()



            # measure accuracy and record loss
            result = Result()
            result.evaluate(pred.data, target.data)

            average_meter.update(result,  input.size(0))
            if args.evaluate:
                val_imgs_metric_saver(result)
            rgb=input
            if args.evaluate :
                img_merge = merge_into_row(rgb * 255, target.cpu(), pred.cpu())
                save_image(img_merge, saved_img_path + '/pred_depth_{:05d}.png'.format(count))

            else:
                skip = 50
                if count == 0:
                    img_merge = merge_into_row(rgb*255, target, pred)
                elif (count < 8 * skip) and (count % skip == 0):
                    row = merge_into_row(rgb*255, target, pred)
                    img_merge = add_row(img_merge, row)
                elif count == 8 * skip:
                    filename = 'comparison_' + str(epoch) + '.png'
                    result_save_path = os.path.join(output_directory, filename)
                    save_image(img_merge, result_save_path)
            count+=1
            val_bar(count)
    avg = average_meter.average()
    epoch_loss = running_loss / len(val_loader)
    print("\tEpoch {} val loss:{:.5f}".format(epoch,epoch_loss) )
    print('RMSE\tMAE\tDelta1\tDelta2\tDelta3\tREL\tLg10\t')
    print('{average.rmse:.3f}\t'
          '{average.mae:.3f}\t'
          '{average.delta1:.3f}\t'
          '{average.delta2:.3f}\t'
          '{average.delta3:.3f}\t'
          '{average.absrel:.3f}\t'
          '{average.lg10:.3f}\t'.format(
        average=avg))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                })
    return avg, img_merge
def main(args):
    global best_result, output_directory, train_csv, test_csv

    if args.loss == 'l2':
        loss_fn = MaskedMSELoss().cuda()
    elif args.loss == 'l1':
        loss_fn = MaskedL1Loss().cuda()
    elif args.loss == 'berhu':
        loss_fn = MaskedberHuLoss().cuda()
    print("loss_fn set.")



    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best models found at '{}'".format(args.evaluate)
        print("=> loading best models '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['models'].cuda()
        print("=> loaded best models (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_dataloaders(args)
        args.evaluate = True

        validate(val_loader, model,loss_fn, checkpoint['epoch'], write_to_file=False)
        return
    # 2.Load models
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['models'].cuda()
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_dataloaders(args)
        args.resume = True

    else:
        train_loader, val_loader = create_dataloaders(args)
        print("=> creating Model FCRN ...")
        model=FCRN(train_loader.dataset.output_size)
        model=model.cuda()
        output_directory = get_output_directory()

        print("=> models created.")

        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        start_epoch = 0
        print("start validate before training")
        best_result, _ = validate(val_loader, model, loss_fn, 'val_before_train', write_to_file=False)

    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    if not (args.resume ):
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch,args.epochs):
        print("=" * 30, "  epoch [{}/{}] ".format(epoch,args.epochs), "=" * 30)

        train(train_loader, model, loss_fn, optimizer, epoch)
        result, img_merge = validate(val_loader, model, loss_fn, epoch)

        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result

            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\ndelta2={:.3f}\ndelta3={:.3f}\n\n".
                        format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1,result.delta2,
                              result.delta3,))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                save_image(img_merge, img_filename)

            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'models': model,
                'best_result': best_result,
                'optimizer': optimizer,
            }, is_best, epoch, output_directory)
        print("best result rmse:", best_result.rmse)
        print()





if __name__ == '__main__':

    main(args)

