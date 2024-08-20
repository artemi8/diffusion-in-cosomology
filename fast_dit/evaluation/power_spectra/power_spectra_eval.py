import numpy as np
import Pk_library as PKL
import matplotlib.pyplot as plt
import plotext as pltxt
import os
import argparse

def get_power_spectra(file, box_size, MAS, array_in=False, print_outs=False):
    
    if array_in:
        if isinstance(file, np.ndarray):
            delta = file
        else:
            raise Exception(f"Invalid object type: {type(file)}. Expected a NumPy array (numpy.ndarray).")
    else:
        extension = os.path.splitext(file)[-1].lower()
        if extension != '.npy':
            raise Exception(f"Invalid file extension '{extension}'. Expected a '.npy' file.")
        delta = np.load(file)

    BoxSize = box_size #1000.0  #Mpc/h
    MAS     = MAS #MAS used to create the image; 'NGP', 'CIC', 'TSC', 'PCS' o 'None'
    threads = 1      #number of openmp threads


    # compute the Pk of that image
    Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads)

    # get the attributes of the routine
    k      = Pk2D.k      #k in h/Mpc
    Pk     = Pk2D.Pk     #Pk in (Mpc/h)^2
    Nmodes = Pk2D.Nmodes #Number of modes in the different k bins
    
    if print_outs:
        print(f'k : {k} h/Mpc')
        print(f'Pk : {Pk} (Mpc/h)^2')
        print(f'Nmodes : {Nmodes} Number of modes in the different k bins')
    
    return k, Pk, Nmodes

def compute_average_power_spectrum(image_paths, box_size, MAS):
    all_Pk = []
    for image_path in image_paths:
        k, Pk, _ = get_power_spectra(image_path, box_size, MAS)
        all_Pk.append(Pk)
    
    # Convert list to numpy array and compute the mean across all images
    all_Pk = np.array(all_Pk)
    mean_Pk = np.mean(all_Pk, axis=0)
    
    return k, mean_Pk


def compare_power_spectra(image_path1, image_path2, box_size, MAS, array_in=False,
                           single_eval=True, plot_save_path='./spectrum_comparison.jpg',
                             terminal_out=True):
    
    k1, Pk1, _ = get_power_spectra(image_path1, box_size, MAS, array_in=array_in)
    k2, Pk2, _ = get_power_spectra(image_path2, box_size, MAS, array_in=array_in)

    if single_eval == True:
        # Mean squared Error
        mse = np.mean(np.square(Pk1 - Pk2))
        print(f'Mean Squared Error : {mse}')

        # Plot using plotext for terminal output
    if terminal_out:
        pltxt.plot(k1, Pk1, label="Original")
        pltxt.plot(k2, Pk2, label="Generated")
        pltxt.title("Power Spectra Comparison (Terminal)")
        pltxt.xlabel("k (h/Mpc)")
        pltxt.ylabel("Power Spectrum P(k) [(Mpc/h)^2]")
        pltxt.show()
    
    if plot_save_path:
        plt.figure()
        plt.loglog(k1, Pk1, label='Original')
        plt.loglog(k2, Pk2, label='Generated', linestyle='--')
        plt.xlabel('k (h/Mpc)')
        plt.ylabel('Power Spectrum P(k) [(Mpc/h)^2]')
        plt.legend()
        plt.savefig(plot_save_path)
        if not terminal_out:
            plt.show()
        plt.close()
        


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Get Power Spectra Comparison for two numpy arrays')
        parser.add_argument('--image_path_1', type=str, required=True, help='Path to the Artificially Generated array')
        parser.add_argument('--image_path_2', type=str, required=True, help='Path to the Original array')
        parser.add_argument('--box_size', type=int, required=True, help='Box size of the simuation it is sliced from')
        parser.add_argument('--MAS', type=str, required=True, help='Mass Assignment Scheme used in while fetching the simulated data')
        args = parser.parse_args()
        compare_power_spectra(image_path1 = args.image_path_1,
                              image_path2 = args.image_path_2,
                              box_size = args.box_size,
                              MAS = args.MAS)