import argparse

loss_names = ['l1', 'l2','berhu']

parser = argparse.ArgumentParser(description='PyTorch Depth Map Prediction')
parser.add_argument('--gpu', type=str,default="0", metavar='N',
                    help='gpu id ')

parser.add_argument('--epochs', type=int, default = 20, metavar='N',
                    help='number of epochs to train(default: 20) ')

parser.add_argument('--batch_size','--b',type=int,default=32,metavar="batch",
                    help='input batch size for training(default: 32) ')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of Data loading workers (default: 16)')
parser.add_argument("--lr",'--learning-rate',type=float,default=1e-4,metavar='learning rate',
                    help="initial learning rate (default 1e-4)")


parser.add_argument("--weight_decay",'--wd',type=float,default=1e-4,metavar="wd",
                    help="weight decay(default: 1e-4)")
parser.add_argument( '--loss', metavar='LOSS', default='berhu', choices=loss_names,
                    help='loss function: ' + ' | '.join(loss_names) + ' (default: berhu)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                    help='evaluate models on validation set')


args = parser.parse_args()

