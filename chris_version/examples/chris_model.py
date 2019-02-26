import torch
import torch.optim

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, F_conv

import chris_data as data

from sys import exit

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)

def fit(input, target):
    return torch.mean((input - target)**2)

def train(model,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0

    t_start = time()

    loss_factor = 600**(float(i_epoch) / 300) / 600
    if loss_factor > 1:
        loss_factor = 1

    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        y_clean = y.clone()
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))


        optimizer.zero_grad()

        # Forward step:

        model = model.cuda()
        output = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])

        output_block_grad = torch.cat((output[:, :ndim_z],
                                       output[:, -ndim_y:].data), dim=1)

        l += lambd_latent * loss_latent(output_block_grad, y_short)
        l_tot += l.data.item()

        l.backward()

        # Backward step:
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                y), dim=1)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        l_rev = (
            lambd_rev
            * loss_factor
            * loss_backward(output_rev_rand[:, :ndim_x],
                            x[:, :ndim_x])
        )

        l_rev += 0.50 * lambd_predict * loss_fit(output_rev, x)

        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in model.parameters():
            p.grad.data.clamp_(-15.00, 15.00)

        optimizer.step()

        #     print('%.1f\t%.5f' % (
        #                              float(batch_idx)/(time()-t_start),
        #                              l_tot / batch_idx,
        #                            ), flush=True)

    return l_tot / batch_idx

def main():

    # Set up simulation parameters
    batch_size = 1600  # set batch size
    r = 4              # the grid dimension for the output tests
    test_split = r*r   # number of testing samples to use
    sigma = 0.2        # the noise std
    ndata = 64         # number of data samples
    usepars = [0,1,2,3]    # parameter indices to use
    seed = 1           # seed for generating data
    run_label='gpu0'
    out_dir = "/home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/multipar/%s/" % run_label

    # generate data
    pos, labels, x, sig, parnames = data.generate(
        tot_dataset_size=2**20,
        ndata=ndata,
        usepars=usepars,
        sigma=sigma,
        seed=seed
    )
    print('generated data')

    # seperate the test data for plotting
    pos_test = pos[-test_split:]
    labels_test = labels[-test_split:]
    sig_test = sig[-test_split:]

    # plot the test data examples
    plt.figure(figsize=(6,6))
    fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')
    cnt = 0
    for i in range(r):
        for j in range(r):
            axes[i,j].plot(x,np.array(labels_test[cnt,:]),'.')
            axes[i,j].plot(x,np.array(sig_test[cnt,:]),'-')
            cnt += 1
            axes[i,j].axis([0,1,-1.5,1.5])
            axes[i,j].set_xlabel('time') if i==r-1 else axes[i,j].set_xlabel('')
            axes[i,j].set_ylabel('h(t)') if j==0 else axes[i,j].set_ylabel('')
    plt.savefig('%stest_distribution.png' % out_dir,dpi=360)
    plt.close()

    # precompute true posterior samples on the test data
    cnt = 0
    N_samp = 1000
    ndim_x = len(usepars)
    samples = np.zeros((r*r,N_samp,ndim_x))
    for i in range(r):
        for j in range(r):
            samples[cnt,:,:] = data.get_lik(np.array(labels_test[cnt,:]).flatten(),sigma=sigma,usepars=usepars,Nsamp=N_samp)
            print(samples[cnt,:10,:])
            cnt += 1

    # initialize plot for showing testing results
    fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')

    for k in range(ndim_x):
        parname1 = parnames[k]
        for nextk in range(ndim_x):
            parname2 = parnames[nextk]
            if nextk>k:
                cnt = 0
                for i in range(r):
                     for j in range(r):

                         # plot the samples and the true contours
                         axes[i,j].clear()
                         axes[i,j].scatter(samples[cnt,:,k], samples[cnt,:,nextk],c='b',s=0.5,alpha=0.5)
                         axes[i,j].plot(pos_test[cnt,k],pos_test[cnt,nextk],'+c',markersize=8)
                         axes[i,j].set_xlim([0,1])
                         axes[i,j].set_ylim([0,1])
                         axes[i,j].set_xlabel(parname1) if i==r-1 else axes[i,j].set_xlabel('')
                         axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
                         
                         cnt += 1

                # save the results to file
                fig.canvas.draw()
                plt.savefig('%strue_samples_%d%d.png' % (out_dir,k,nextk),dpi=360)

    # setting up the model 
    ndim_x = len(usepars)        # number of posterior parameter dimensions (x,y)
    ndim_y = ndata    # number of label dimensions (noisy data samples)
    ndim_z = 4        # number of latent space dimensions?
    ndim_tot = max(ndim_x,ndim_y+ndim_z)     # must be > ndim_x and > ndim_y + ndim_z

    # define different parts of the network
    # define input node
    inp = InputNode(ndim_tot, name='input')

    # define hidden layer nodes
    t1 = Node([inp.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    #t1 = Node([inp.out0], rev_multiplicative_layer,
    #          {'F_class': F_conv, 'clamp': 2.0,
    #           'F_args': {'kernel_size': 3,'leaky_slope': 0.1}})

    #def __init__(self, dims_in, F_class=F_fully_connected, F_args={},
    #             clamp=5.):
    #    super(rev_multiplicative_layer, self).__init__()
    #    channels = dims_in[0][0]
    #
    #    self.split_len1 = channels // 2
    #    self.split_len2 = channels - channels // 2
    #    self.ndims = len(dims_in[0])
    #
    #    self.clamp = clamp
    #    self.max_s = exp(clamp)
    #    self.min_s = exp(-clamp)
    #
    #    self.s1 = F_class(self.split_len1, self.split_len2, **F_args)
    #    self.t1 = F_class(self.split_len1, self.split_len2, **F_args)
    #    self.s2 = F_class(self.split_len2, self.split_len1, **F_args)
    #    self.t2 = F_class(self.split_len2, self.split_len1, **F_args)

    t2 = Node([t1.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    t3 = Node([t2.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.2}})

    t4 = Node([t3.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.0}})

    # define output layer node
    outp = OutputNode([t4.out0], name='output')

    nodes = [inp, t1, t2, t3, t4, outp]
    model = ReversibleGraphNet(nodes)

    # Train model
    # Training parameters
    n_epochs = 10000
    meta_epoch = 12 # what is this???
    n_its_per_epoch = 12
    batch_size = 1600

    lr = 1e-2
    gamma = 0.01**(1./120)
    l2_reg = 2e-5

    y_noise_scale = 3e-2
    zeros_noise_scale = 3e-2

    # relative weighting of losses:
    lambd_predict = 300. # forward pass
    lambd_latent = 300.  # laten space
    lambd_rev = 400.     # backwards pass

    # padding both the data and the latent space
    # such that they have equal dimension to the parameter space
    #pad_x = torch.zeros(batch_size, ndim_tot - ndim_x)
    #pad_yz = torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.8),
                             eps=1e-04, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=meta_epoch,
                                                gamma=gamma)

    # define the three loss functions
    loss_backward = MMD_multiscale
    loss_latent = MMD_multiscale
    loss_fit = fit

    # set up training set data loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
        batch_size=batch_size, shuffle=True, drop_last=True)

    # initialisation of network weights
    #for mod_list in model.children():
    #    for block in mod_list.children():
    #        for coeff in block.children():
    #            coeff.fc3.weight.data = 0.01*torch.randn(coeff.fc3.weight.shape)
    #model.to(device)

    # start training loop            
    try:
        t_start = time()
        olvec = np.zeros((r,r,int(n_epochs/10)))
        s = 0
        # loop over number of epochs
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):

            scheduler.step()

            # Initially, the l2 reg. on x and z can give huge gradients, set
            # the lr lower for this
            if i_epoch < 0:
                print('inside this iepoch<0 thing')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 1e-2

            # train the model
            train(model,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,
                ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,
                loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,i_epoch)

            # loop over a few cases and plot results in a grid
            if np.remainder(i_epoch,10)==0:
                for k in range(ndim_x):
                    parname1 = parnames[k]
                    for nextk in range(ndim_x):
                        parname2 = parnames[nextk]
                        if nextk>k:
                            cnt = 0

                            # initialize plot for showing testing results
                            fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')

                            for i in range(r):
                                for j in range(r):

                                    # convert data into correct format
                                    y_samps = np.tile(np.array(labels_test[cnt,:]),N_samp).reshape(N_samp,ndim_y)
                                    y_samps = torch.tensor(y_samps, dtype=torch.float)
                                    y_samps += y_noise_scale * torch.randn(N_samp, ndim_y)
                                    y_samps = torch.cat([torch.randn(N_samp, ndim_z), zeros_noise_scale * 
                                        torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                                        y_samps], dim=1)
                                    y_samps = y_samps.to(device)

                                    # use the network to predict parameters
                                    rev_x = model(y_samps, rev=True)
                                    rev_x = rev_x.cpu().data.numpy()

                                    # compute the n-d overlap
                                    if k==0 and nextk==1:
                                        ol = data.overlap(samples[cnt,:,:ndim_x],rev_x[:,:ndim_x])
                                        olvec[i,j,s] = ol                                     

                                    # plot the samples and the true contours
                                    axes[i,j].clear()
                                    axes[i,j].scatter(samples[cnt,:,k], samples[cnt,:,nextk],c='b',s=0.2,alpha=0.5)
                                    axes[i,j].scatter(rev_x[:,k], rev_x[:,nextk],c='r',s=0.2,alpha=0.5)
                                    axes[i,j].plot(pos_test[cnt,k],pos_test[cnt,nextk],'+c',markersize=8)
                                    axes[i,j].set_xlim([0,1])
                                    axes[i,j].set_ylim([0,1])
                                    oltxt = '%.2f' % olvec[i,j,s]
                                    axes[i,j].text(0.90, 0.95, oltxt,
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        transform=axes[i,j].transAxes)
                                    matplotlib.rc('xtick', labelsize=8)     
                                    matplotlib.rc('ytick', labelsize=8) 
                                    axes[i,j].set_xlabel(parname1) if i==r-1 else axes[i,j].set_xlabel('')
                                    axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
                                    cnt += 1

                            # save the results to file
                            fig.canvas.draw()
                            plt.savefig('%sposteriors_%d%d_%04d.png' % (out_dir,k,nextk,i_epoch),dpi=360)
                            plt.savefig('%slatest_%d%d.png' % (out_dir,k,nextk),dpi=360)
                            plt.close()
                s += 1

            # plot overlap results
            if np.remainder(i_epoch,10)==0:                
                fig, axes = plt.subplots(1,figsize=(6,6))
                for i in range(r):
                    for j in range(r):
                        axes.semilogx(10*np.arange(olvec.shape[2]),olvec[i,j,:],alpha=0.5)
                axes.grid()
                axes.set_ylabel('overlap')
                axes.set_xlabel('epoch')
                axes.set_ylim([0,1])
                plt.savefig('%soverlap.png' % out_dir,dpi=360)
                plt.close()

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")

main()
