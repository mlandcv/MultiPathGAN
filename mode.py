from networks import Generator
from networks import Discriminator
from networks import FeatureResNet34

import os
import time
import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image

#solver for training and testing 

class MPG_solver():
    def __init__(self, loader, args):
        self.loader = loader
        # generator arguments
        self.G_conv_dim = args.G_conv_dim
        self.num_domains = args.num_domains
        self.G_Resblocks = args.num_G_Resblocks
        # discriminator arguments
        self.img_size = args.img_size
        self.D_conv_dim = args.D_conv_dim
        self.D_Convblocks = args.num_D_Convblocks
        self.d_learning_rate = args.D_learning_rate
        #optimizer arguments
        self.g_learning_rate = args.G_learning_rate
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        #directories
        self.model_dir = args.model_dir
        self.sample_dir = args.sample_dir
        self.log_dir = args.log_dir
        self.result_dir = args.result_dir
        self.target_dir = args.target_dir
        #train arguments
        self.dataset = args.dataset_name
        self.resume_training = args.resume_training
        self.iters = args.train_iters
        self.d_g_steps = args.d_g_steps
        self.lambda_cls_loss = args.lambda_cls_loss
        self.lambda_gp_loss = args.lambda_gp_loss
        self.lambda_recon = args.lambda_recon
        self.lambda_p = args.lambda_p
        #print and save
        self.logstep = args.logstep
        self.save_model_step = args.save_model_step
        self.use_tensorboard = args.use_tensorboard
        self.save_sample_step = args.save_sample_step
        #tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model()
        if self.use_tensorboard:
            self.build_tensorboard()
        #update learning rate decay
        self.update_lr_step = args.update_lr_step
        self.iter_decay = args.iter_decay
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        #test arguments
        self.test_iters = args.test_iters
        #sample arguments
        self.which_domain = args.which_domain

    def train(self):
        print('...Getting Data Batches...')
        data_loader = self.loader

        '''get batch and corresponding labels'''
        data_iter = iter(data_loader)
        x_data, domain = next(data_iter)
        
        print('Mini batch image data shape:', x_data.shape)

        x_data = x_data.to(self.device)
        domain_list = self.get_labels(domain, self.num_domains)

        #print(domain_list)
        #print('mini_batch_domain_list_shape:', np.shape(domain_list))

        begin = 0
        
        #if resume training from prevoius checkpoint
        if self.resume_training:
            begin = self.resume_training
            self.restore_train(self.resume_training)

        #else start from scratch
        print('...Starting Training...')

        start_time= time.time()
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
        
        g_lr = self.g_lr
        d_lr = self.d_lr
        
        #get mini batches
        for i in range(begin, self.iters):
            data_iter = iter(data_loader)
            real_x, real_lbl = next(data_iter)
            
            # Randomly Generate fake  target labels 
            random_trg = torch.randperm(real_lbl.size(0))
            tgt_lbl = real_lbl[random_trg]

            #convert to onehot
            onehot_real_lbl = self.label_to_onehot(real_lbl, self.num_domains)
            onehot_tgt_lbl = self.label_to_onehot(tgt_lbl, self.num_domains)

            #print(real_x)
            #print(real_lbl)
            #print(tgt_lbl)
            #print(onehot_real_lbl)
            #print(onehot_tgt_lbl)
            
            #real_x-->input image
            #real_lbl-->real labels
            #tgt_lbl-->randomly created target labels
            #onehot_real_lbl-->one hot real labels
            #onehot_tgt_lbl-->random fake onehot target labels

            #To GPU
            real_x = real_x.to(self.device)
            real_lbl = real_lbl.to(self.device)
            tgt_lbl = tgt_lbl.to(self.device)
            onehot_real_lbl = onehot_real_lbl.to(self.device)
            onehot_tgt_lbl = onehot_tgt_lbl.to(self.device)

            #TRAIN DISCRIMINATOR
            #real image loss
            out_src, out_cls = self.D(real_x)
            real_dloss = -torch.mean(out_src)
            cls_dloss = F.cross_entropy(out_cls, real_lbl)

            #fake image loss
            fake_x = self.G(real_x, onehot_tgt_lbl)
            out_src, out_cls = self.D(fake_x.detach())
            fake_dloss = torch.mean(out_src)

            #gradient penalty
            alpha = torch.rand(real_x.size(0),1,1,1).to(self.device)
            x_hat = (alpha * real_x.data + (1- alpha) * fake_x.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            gp_dloss = self.gradient_penalty(out_src, x_hat)

            #backprop
            loss_d = real_dloss + fake_dloss + self.lambda_cls_loss*cls_dloss + self.lambda_gp_loss*gp_dloss            
            self.reset_gradients()
            loss_d.backward()
            self.D_optimizer.step()

            #logs
            loss = {}
            loss['D/real_loss'] = real_dloss.item()
            loss['D/fake_loss'] = fake_dloss.item()
            loss['D/cls_dloss'] = cls_dloss.item()
            loss['D/gp_dloss'] = gp_dloss.item()

            #TRAIN GENERATOR

            if (i+1) % self.d_g_steps == 0:
                #domain a --> domain b
                fake_x = self.G(real_x, onehot_tgt_lbl)
                out_src, out_cls = self.D(fake_x)
                adv_loss = -torch.mean(out_src)
                cls_loss = F.cross_entropy(out_cls, tgt_lbl)

                #domain b --> domain a
                cycrecon_x = self.G(fake_x, onehot_real_lbl)
                recon_loss = torch.mean(torch.abs(real_x - cycrecon_x))

                #ADD PERCEPTION LOSS

                #self.gpu_ids = 4

                featrealA = self.F(real_x)  #.cuda() if required
                featfakeB = self.F(fake_x)  #.cuda() if required
                #featrealB = self.netfeat(x_fake)
                #featfakeA = self.F(cycrecon_x)

                #mse loss equation
                p_loss_AfB = torch.sum((featrealA - featfakeB)**2)/featrealA.data.nelement()
                #p_loss_BfA = self.mse_loss(featrealB, featfakeA)
                #p_loss = feat_loss_AfB + feat_loss_BfA
                p_loss = p_loss_AfB

                #BACKPROPAGATION
                totalG_loss = adv_loss + self.lambda_recon * recon_loss + self.lambda_cls_loss * cls_loss + self.lambda_p * p_loss
                self.reset_gradients()
                totalG_loss.backward()
                self.G_optimizer.step()

                       ###########  E  N  D  #############

                #save logs
                loss['G/adv_loss'] = adv_loss.item()
                loss['G/recon_loss'] = recon_loss.item()
                loss['G/cls_loss'] = cls_loss.item()
                loss['G/p_loss'] = p_loss.item()

            #print and tensorboard log 
            if (i+1) % self.logstep == 0:
                elap_time = time.time() - start_time
                elap_time = str(datetime.timedelta(seconds=elap_time))[:-7]
                log = "Elapsed time[{}], Iteration number[{}/{}]".format(elap_time, i+1, self.iters)

                for loss_name, value in loss.items():
                    log += ", {}: {:.4f}".format(loss_name, value)
                print(log)

                if self.use_tensorboard:
                    for loss_name, value in loss.items():
                        self.logger.scalar_summary(loss_name, value, i+1)

            # Saving checkpoints.
            if (i+1) %  self.save_model_step == 0:
                pathG = os.path.join(self.model_dir,'{}-G.ckpt'.format(i+1))
                pathD = os.path.join(self.model_dir,'{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), pathG)
                torch.save(self.D.state_dict(), pathD)
                print('Checkpoint model saved in {}...'.format(self.model_dir))

            # sample results from current checkpoint
            if (i+1) % self.save_sample_step == 0:
                with torch.no_grad():
                    x_data_list = [x_data]

                    for dmain in domain_list:
                        x_data_list.append(self.G(x_data, dmain))
                    x_concat = torch.cat(x_data_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            #learning rate decay
            if (i+1) % self.update_lr_step == 0 and (i+1) > (self.iters - self.iter_decay):
                g_lr -= (self.g_lr / float(self.iter_decay))
                d_lr -= (self.d_lr / float (self.iter_decay))
                self.update_lr(g_lr,d_lr)
                print ('Decayed learning rates, G_lr: {}, D_lr: {}.'.format(g_lr, d_lr))



    def test(self):
        data_loader = self.loader

        '''image translation'''
        print('Loading trained generator')
        self.restore_train(self.test_iters)

        with torch.no_grad():
            for i, (real_x, real_lbl) in enumerate(data_loader):
                real_x = real_x.to(self.device)
                trg_domain_list = self.get_labels(real_lbl, self.num_domains)
            fake_x_list = [real_x]

            for trg_domain in trg_domain_list:
                '''translate'''
                fake_x_list.append(self.G(real_x, trg_domain))
            '''save image'''
            x_concat = torch.cat(fake_x_list, dim=3)
            res_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), res_path, nrow=1, padding=0)
            print('Saving real and fake images into folder path {}...'.format(res_path))



    def sample(self):
        '''image translation'''
        print('Loading trained generator')
        self.restore_train(self.test_iters)
        '''load data'''
        data_loader = self.loader
        with torch.no_grad():
            for i, (real_x, real_lbl) in enumerate(data_loader):
                real_x = real_x.to(self.device)
                trg_domain_list = self.get_labels(real_lbl, self.num_domains)
                '''translate'''
                for j, trg_domain in enumerate(trg_domain_list):
                    x_fake = self.G(real_x, trg_domain)
                    
                    '''save image'''
                    for k in range (len(x_fake)):
                        if j == self.which_domain:
                            sample_path = os.path.join(self.target_dir, f'fake_domain{self.which_domain}_{i+1}.png')
                            save_image(self.denorm(x_fake.data[k].cpu()), sample_path, nrow=1, padding=0)
                            print('Saving translated images into folder path {}...'.format(sample_path))


                        


    def get_labels(self, domain, num_domains):
        '''get domain labels'''

        trg_domain_list = []

        for i in range(num_domains):
            #print('mini batch vector of domain labels:',torch.ones(domain.size(0))*i)
            trg_domains = self.label_to_onehot(torch.ones(domain.size(0))*i, num_domains)
            trg_domain_list.append(trg_domains.to(self.device))
        
        return trg_domain_list

    def label_to_onehot(self, data_labels, data_dim):
        '''labels to one-hot vectors'''
        batch = data_labels.size(0)
        #print('batch:',batch)
        output = torch.zeros(batch, data_dim)
        #print('output1_shape:',output.shape)
        output[np.arange(batch), data_labels.long()] = 1


        #print('output:',output)
        #print('output_shape:',output.shape)

        return output



    def model(self):
        self.G = Generator(self.G_conv_dim, self.num_domains, self.G_Resblocks)
        self.D = Discriminator(self.img_size, self.D_conv_dim, self.num_domains, self.D_Convblocks)
        #self.F = FeatureResNet34()
        self.F = FeatureResNet34(gpu_ids = None).cuda() #check this out if there is an error with the GPU

        self.G_optimizer = torch.optim.Adam(self.G.parameters(), self.g_learning_rate, [self.beta_1, self.beta_2])
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), self.d_learning_rate, [self.beta_1, self.beta_2])
        

        self.print_network(self.G, 'Generator')
        self.print_network(self.D, 'Discriminator')
        self.print_network(self.F, 'Feature extractor ResNet')

        self.G.to(self.device)
        self.D.to(self.device)

        #self.F.to(self.device)

    def print_network(self, network, network_name):
        '''Print out the network information.'''
        num_params = 0
        for p in network.parameters():
            num_params += p.numel()
        print(network)
        print(network_name)
        print("The number of parameters for %s: {}".format(num_params) % (network_name))


    def restore_train(self, which_step):
        '''restore training from previous model/step'''
        print('Loading trained model from iter number{}---'.format(which_step))
        G_model_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(which_step))
        D_model_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(which_step))
        self.G.load_state_dict(torch.load(G_model_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_model_path, map_location=lambda storage, loc: storage))
        
    def reset_gradients(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.G_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.D_optimizer.param_groups:
            param_group['lr'] = d_lr    

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)