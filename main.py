import argparse
from misc import loader
from torch.backends import cudnn

from mode import MPG_solver

def main(args):
    cudnn.benchmark = True
    
    if args.mode == 'train':
        train_loader = loader(args.train_directory, 
                                args.mode, args.img_crop, 
                                args.img_size, args.dataset_name, 
                                args.batch_size, args.num_workers)
        mpg_solver_train = MPG_solver(train_loader, args)
        mpg_solver_train.train()

    elif args.mode == 'test':
        test_loader = loader(args.test_directory, 
                                args.mode, args.img_crop, 
                                args.img_size, args.dataset_name, 
                                args.batch_size, args.num_workers)
        mpg_solver_test = MPG_solver(test_loader, args)
        mpg_solver_test.test()

    elif args.mode == 'sample':
        sample_loader = loader(args.input_sample_directory, 
                                args.mode, args.img_crop, 
                                args.img_size, args.dataset_name, 
                                args.batch_size, args.num_workers)
        mpg_solver_sample = MPG_solver(sample_loader, args)
        mpg_solver_sample.sample()

def str2bool(v):
    return v.lower() in ('true')

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    #directory arguments
    #input
    parser.add_argument('--train_directory', type=str, default='data/wsi/train')
    parser.add_argument('--test_directory', type=str, default='data/wsi/test')
    parser.add_argument('--input_sample_directory', type=str, default='data/wsi/input_sample_dir')
    #output
    parser.add_argument('--model_dir', type=str, default='multipathgan/models')
    parser.add_argument('--sample_dir', type=str, default='multipathgan/sample')
    parser.add_argument('--log_dir', type=str, default='multipathgan/logs')
    parser.add_argument('--result_dir', type=str, default='multipathgan/results')
    parser.add_argument('--target_dir', type=str, default='multipathgan/target_samples')

    #image data loader arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train','test','sample'])
    parser.add_argument('--img_crop', type= int, default = 256, help='center crop size of image if image is greater than 256x256')
    parser.add_argument('--img_size', type=int, default = 256, help='Input/Output image resolution')
    parser.add_argument('--dataset_name', type=str, default='wsi', choices=['wsi','vidit']) #add dataset name to choices if you want to save it with a different name
    parser.add_argument('--batch_size', type=int, default=16, help="mini-batch size for training")
    parser.add_argument('--num_workers', type=int, default=1)

    #model arguments
    #generator 
    parser.add_argument('--G_conv_dim', type=int, default=64, help='number of first layer conv filters in G')
    parser.add_argument('--num_domains', type=int, default=3, help='number of domains labels/classes in the dataset')
    parser.add_argument('--num_G_Resblocks', type=int, default=6, help='number of Res blocks in G backbone' )
    parser.add_argument('--G_learning_rate', type=float, default=0.0001, help='Generator learning rate')
    #discriminator
    parser.add_argument('--D_conv_dim', type=int, default=64, help='number of first layer conv filters in D')
    parser.add_argument('--num_D_Convblocks', type=int, default=6, help='number of strided convs in D' )
    parser.add_argument('--D_learning_rate', type=float, default=0.0001, help='learning rate for D')
    #optimizer
    parser.add_argument('--beta_1', type=int, default=0.5, help='adam optimizer beta1')
    parser.add_argument('--beta_2', type=int, default=0.999, help='adam optimizer beta2')
    
    #training
    parser.add_argument('--resume_training', type=int, default=None, help='resume training from previous saved model/step')
    parser.add_argument('--train_iters', type=int, default=200000, help='total number of training iterations')
    parser.add_argument('--d_g_steps', type=int, default=5, help='number of D updates for each G update')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--update_lr_step', type=int, default=1000, help='update learning rate every 1000 steps')
    parser.add_argument('--iter_decay', type=int, default=100000, help='iter number before which we decay learning rate')
    
    #loss weight hyperparameters
    parser.add_argument('--lambda_cls_loss', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_gp_loss', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_recon', type=float, default=10, help='weight for cycle reconstruction loss')
    parser.add_argument('--lambda_p', type=float, default=1, help='weight for perception loss')

    #printing and model saving
    parser.add_argument('--logstep', type=int, default=10, help='saves logs after every 10 steps')
    parser.add_argument('--save_model_step', type=int, default=10000, help='saves model checkpoints after every 10000 steps')
    parser.add_argument('--save_sample_step', type=int, default=1000, help='saves sample results checkpoints after every 1000 steps')
    parser.add_argument('--use_tensorboard', type=str2bool, default=False, help='update tensorboard')

    #test model
    parser.add_argument('--test_iters', type=int, default=200000, help='test model at step 200000')
    parser.add_argument('--which_domain', type=int, default=0, help='choose target domain for test set between "0" and "total number of domains"')
    
    
    args = parser.parse_args()
    print(args)
    main(args)

