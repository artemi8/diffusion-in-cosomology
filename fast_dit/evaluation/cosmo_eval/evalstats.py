import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from numpy.fft import *
from quantimpy import minkowski as mk
import argparse

def gen_samplelist(datapath_kwargs:dict, num_samples, power_spectrum=False):
    
    real_sample_dir = datapath_kwargs['real_sample_dir']
    gen_sample_dir = datapath_kwargs['gen_sample_dir']
    
    real_arrays = []
    gen_arrays = []
        
    for img in random.sample(os.listdir(real_sample_dir), k=num_samples):
        if img.endswith('.png'):
            temp_img = Image.open(os.path.join(real_sample_dir, img)).convert('RGB')
            if power_spectrum:
                temp_img = np.mean(np.array(temp_img), axis=2)
            real_arrays.append(temp_img)
            
    for img in random.sample(os.listdir(gen_sample_dir), k=num_samples):
        if img.endswith('.png'):
            temp_img = Image.open(os.path.join(gen_sample_dir, img)).convert('RGB')
            if power_spectrum:
                temp_img = np.mean(np.array(temp_img), axis=2)
            gen_arrays.append(temp_img)
            
    samplist = [np.array(real_arrays), np.array(gen_arrays)]
    
    return samplist
def get_pixel_histogram_for_samples(samplist, hist_kwargs, names, cols, with_err=True, savefig_dict={}):
    '''
    :param datapath_kwargs: dict with keys real_sample_dir, gen_sample_dir
                             which specifies path to sample and original images
    :param hist_kwargs: bins, range, density
    :return:
    '''
    
    sampwise_histmean  = []
    sampwise_histstd = []
    for samp in samplist:
        hist_all = np.zeros((samp.shape[0], len(hist_kwargs['bins'])-1))
        for img in range(samp.shape[0]):
            vals = np.histogram(samp[img][:], **hist_kwargs)
            hist_all[img, :] = vals[0]
        sampwise_histmean.append(hist_all.mean(0))
        sampwise_histstd.append(np.std(hist_all, axis=0, ddof=1))
    #bins
    bins = hist_kwargs['bins']
    bins_low = bins[:-1]
    bins_upp = bins[1:]
    bins_mid = (bins_upp+bins_low)/2
    bins_width = bins_upp - bins_low
    plt.figure()
    for isa in range(len(samplist)):
        if with_err:
            plt.bar(bins_mid, sampwise_histmean[isa], yerr=sampwise_histstd[isa]/np.sqrt(len(samplist[0])),width=bins_width,
                 label=names[isa], color=cols[isa], alpha=0.2, ecolor=cols[isa])
        else:
            plt.bar(bins_mid, sampwise_histmean[isa], width=bins_width,
                 label=names[isa], color=cols[isa], alpha=0.2)
    plt.legend()
    plt.xlabel('Pixel intensity')
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return sampwise_histmean, sampwise_histstd


def plot_ps_samples(kvals, samplist, names, cols=['b', 'r'], logscale=True, k2pk=False, savefig_dict={}):
    '''
    :param kvals:
    :param samplist: List of power spectra for samples (eg: either from different models or the real fields)
    :param names:
    :param cols:
    :return:
    '''
    
    plt.figure(figsize=savefig_dict['figsize'] if 'figsize' in savefig_dict.keys() else [6, 6])
    for isd, samp in enumerate(samplist):
        assert len(samp.shape)==2
        if k2pk:
            samp = samp*(kvals**2) #check this line
        meanps = np.mean(samp, axis=0)
        stdps = np.std(samp, axis=0, ddof=1)
        style='solid' if isd==0 else 'dashed'
        plt.plot(kvals, meanps, c=cols[isd], label=names[isd], linestyle=style)
        plt.fill_between(kvals, meanps-stdps, meanps+stdps, alpha=0.2, color=cols[isd])
    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.xlabel(r'k')
    if k2pk:
        plt.ylabel(r'$k^2P(k)$')
    else:
        plt.ylabel(r'P(k)')
    plt.legend()
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return


def get_powspec_for_samples(samplist):
    '''
    :param samplist: list of np arrays with shape N_img, Nx, Nx
    :param hist_kwargs: bins, range, density
    :return:
    '''
    ps_list = []
    Nx = samplist[0].shape[-1]
    kvals = np.arange(0, Nx/2)
    for samp in samplist:
        assert len(samp.shape)==3
        assert samp.shape[-1]==Nx
        assert samp.shape[-2]==Nx
        pssamp = np.vstack([calc_1dps_img2d(kvals, samp[ci, ...], to_plot=False, smoothed=0.25) for ci in range(samp.shape[0])])
        ps_list.append(pssamp)
    return kvals, ps_list

def calc_1dps_img2d(kvals, img, to_plot=True, smoothed=0.5):
    Nx = img.shape[0]
    fft_zerocenter = fftshift(fft2(img)/Nx**2) #Aug
    impf = abs(fft_zerocenter) ** 2.0
    x, y = np.meshgrid(np.arange(Nx), np.arange(Nx))
    R  = np.sqrt((x-(Nx/2))**2+(y-(Nx/2))**2) #Aug
    filt = lambda r: impf[(R >= r - smoothed) & (R < r + smoothed)].mean()
    mean = np.vectorize(filt)(kvals)
    return mean


def plot_mink_functionals(samplist, gs_vals, names, cols, savefig_dict={}):
    sampwise_minkmean  = []
    sampwise_minkstd = []
    for samp in samplist:
        samp_minks = []
        for isa in range(len(samp)):#each image
            image = samp[isa]
            gs_masks = [image>=gs_vals[ig] for ig in range(len(gs_vals))]
            minkowski = []
            for i in range(len(gs_masks)):
                minkowski.append(mk.functionals(gs_masks[i], norm=True))
            minkowski = np.vstack(minkowski) #N_alphax3
            samp_minks.append(minkowski)
        samp_minks = np.stack(samp_minks) #NsampxN_alphax3
        sampwise_minkmean.append(samp_minks.mean(0))
        sampwise_minkstd.append(np.std(samp_minks, axis=0, ddof=1))
    
    fig, ax = plt.subplots(figsize=(10, 15), nrows=3)
    for iax in range(3):
        for isa in range(len(samplist)):
            style='solid' if isa==0 else 'dashed'
            ax[iax].plot(gs_vals, sampwise_minkmean[isa][:, iax], cols[isa], label=names[isa], linestyle=style)
            ax[iax].fill_between(gs_vals, sampwise_minkmean[isa][:, iax]-sampwise_minkstd[isa][:, iax], 
                    sampwise_minkmean[isa][:, iax]+sampwise_minkstd[isa][:, iax], color=cols[isa], alpha=0.2)
        ax[iax].set_xlabel('g')
        if iax==0:
            ax[iax].set_ylabel(r'$\mathcal{M}_{0}(g)$', fontsize=18)
        elif iax==1:
            ax[iax].set_ylabel(r'$\mathcal{M}_{1}(g)$', fontsize=18)
        else:
            ax[iax].set_ylabel(r'$\mathcal{M}_{2}(g)$', fontsize=18)
        if iax==0:
            ax[iax].legend(prop={'size': 20})
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return sampwise_minkmean, sampwise_minkstd


def generate_PIH(samp_list, RANGEMIN, RANGEMAX, save_path):

    bins = np.linspace(RANGEMIN, RANGEMAX, 50)
    sampwise_histmean, sampwise_histstd = get_pixel_histogram_for_samples(
            samplist=samp_list,
            hist_kwargs={'bins': bins, 'density': True}, names=['Real Fields', 'Sampled Fields'], 
            cols=['b', 'r'], with_err=True,
              savefig_dict={'save_path': os.path.join(save_path, 'pix_hist_ckp60.pdf')})
    
def generate_ps(ps_samp_list, save_path):

    kvals, powspeclist = get_powspec_for_samples(ps_samp_list)

    #Saving with log=True and k2pk=False
    plot_ps_samples(kvals, powspeclist, names=['Real Fields', 'Sampled Fields'], 
            savefig_dict={'save_path': os.path.join(save_path, 'log_powspec64_ckp60.png'),
                           'figsize': [5, 4]})
    
    #Saving with log=True and k2pk=True
    plot_ps_samples(kvals, powspeclist, names=['Real Fields', 'Sampled Fields'],
                     k2pk=True, logscale=True,
                    savefig_dict={'save_path': os.path.join(save_path, 'k2pk-log64_ckp60.png'),
                                   'figsize': [5, 4]})
    
    #Saving with log=False and k2pk=True
    plot_ps_samples(kvals, powspeclist, names=['Real Fields', 'Sampled Fields'],
                     k2pk=True, logscale=False,
                     savefig_dict={'save_path': os.path.join(save_path, 'k2pk_64_ckp60.png'),
                                   'figsize': [5, 4]})

def mink_funcs(samp_list, RANGEMIN, RANGEMAX, save_path):

    smm, sms = plot_mink_functionals(samp_list, 
                        gs_vals = np.linspace(RANGEMIN, RANGEMAX, 50),
                        names = ['Real Fields', 'Sampled Fields'], cols = ['b', 'r'], 
                        savefig_dict={'save_path': os.path.join(save_path, 'mink_funcs_ckp60.pdf')})

            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_samples", type=str)
    parser.add_argument("--gen_samples",  type=str)
    parser.add_argument("--num_samples",  type=int, default=100)
    parser.add_argument("--save_path",  type=str)
    parser.add_argument("--pixel_min",  type=int, default=10)
    parser.add_argument("--pixel_max",  type=int, default=150)

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    
    datapath_kwargs = {'real_sample_dir': args.real_samples,
                   'gen_sample_dir' : args.gen_samples}
    
    samp_list = gen_samplelist(datapath_kwargs=datapath_kwargs ,
                           num_samples=args.num_samples, 
                           power_spectrum=False)
    print(f'Generating statistics using following configuation : {args}')

    print('Generating pixel Intesity Histogram')
    generate_PIH(samp_list, args.pixel_min, args.pixel_max, args.save_path)

    print('Generating Minkowski functionals')
    mink_funcs(samp_list, args.pixel_min, args.pixel_max, args.save_path)

    samp_list = gen_samplelist(datapath_kwargs=datapath_kwargs,
                           num_samples=args.num_samples, 
                           power_spectrum=True)
    
    print('Generating Power spectrum')
    generate_ps(samp_list, args.save_path)
    
    