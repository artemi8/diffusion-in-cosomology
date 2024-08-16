import numpy as np
import Pk_library as PKL
from PIL import Image
import argparse

def get_power_spectra(args):
# parameters
    delta = Image.open(args.image_path)
    delta = delta.convert("RGB")
    delta = np.array(delta.resize((256, 256), Image.LANCZOS))

    grid    = 256     #the map will have grid^2 pixels
    BoxSize = args.box_size #1000.0  #Mpc/h
    MAS     = args.MAS #MAS used to create the image; 'NGP', 'CIC', 'TSC', 'PCS' o 'None'
    threads = 1       #number of openmp threads

    

    # create an empty image
    # delta = np.zeros((grid,grid), dtype=np.float32)

    # compute the Pk of that image
    Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads)

    # get the attributes of the routine
    k      = Pk2D.k      #k in h/Mpc
    Pk     = Pk2D.Pk     #Pk in (Mpc/h)^2
    Nmodes = Pk2D.Nmodes #Number of modes in the different k bins
    print(f'k : {Pk2D.k} h/Mpc')
    print(f'Pk : {Pk2D.Pk} (Mpc/h)^2')
    print(f'Nmodes : {Pk2D.Nmodes} Number of modes in the different k bins')

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Get Power Spectra for an Image')
        parser.add_argument('--image_path', type=str, required=True, help='Path to the image')
        parser.add_argument('--box_size', type=int, required=True, help='Box size of the simuation it is sliced from')
        parser.add_argument('--MAS', type=str, required=True, help='Mass Assignment Scheme used in while fetching the simulated data')
        args = parser.parse_args()
        get_power_spectra(args)