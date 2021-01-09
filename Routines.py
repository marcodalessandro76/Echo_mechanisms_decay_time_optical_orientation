from mppi import Utilities as U, Parsers as P
import numpy as np


def eval_pulse_parameters(dipole,field_int,width,verbose=True):
    """
    Compute the coupling frequency and the pulse area in function of the values of the
    transition dipole and of the field intensity.
    
    Args:
        dipole : value of the dipole for the selected kpoint and transition
        field_int : field intensity in kW/cm^2
        width : in fs
    
    Returns:
        (coupling_frequency,theta) : a tuple  with the Rabi coupling frequency 
            (in fs^-1) and the pulse area theta
        
    """
    Z0 = U.vacuum_impedence
    field_int = field_int*1e3*1e4 #W/m^2
    field_amp = np.sqrt(Z0*field_int) #V/m
    field_amp = field_amp*U.Bohr_radius #V/a0 in atomic units
    dip_mod = np.linalg.norm(dipole)
    coupling_frequency = dip_mod*field_amp*2*np.pi/U.Planck_ev_ps*1e-3 #fs^-1
    
    theta = np.sqrt(2*np.pi)*width*coupling_frequency
    if verbose:
        print('coupling frequency (THz):',coupling_frequency*1e3)
        print('pulse area :',theta)
    return coupling_frequency,theta

def eval_field_intensity(dipole,theta,width,verbose=True):
    """
    Compute the field intensity that produce the pulse are given as input.
        
    Args:
        dipole : name of the dipoles database
        theta : pulse area
        width : 
    
    """
    Z0 = U.vacuum_impedence
    dip_mod = np.linalg.norm(dipole)
    coupling_frequency = theta/(np.sqrt(2*np.pi)*width) #fs^-1
    field_amp = coupling_frequency/(dip_mod*2*np.pi/(U.Planck_ev_ps*1e3)) #V/a0 in atomic units
    field_amp = field_amp/U.Bohr_radius #V/m
    field_int = field_amp**2/Z0 #W/m^2
    field_int = field_int*1e-3*1e-4 #kW/cm^2
    if verbose: print('field intensity (kW/cm^2) :',field_int)
    return field_int


####################################################################################

def eval_effective_field_int(width,pump_energy,trans_energy,verbose=False):
    """
    Compute the effective field intensity associated to a specific transition.
    
    Args:
        width : temporal width of the field in fs
        pump_energy : energy of the pump in eV
        trans_energy : energy of the transition in eV
        
    Return:
        (float) : scale factor for the intensity of the field
    """
    hplanck = U.Planck_ev_ps*1e3 # Planck constant in ev*fs
    nupump = pump_energy/hplanck # in fs^-1
    t0 = 3*width # position of the maximum of the pulse
    T = 200*width # lenght of the time interval
    dt = 0.1 # resolution of time sampling
    N = int(T/dt) # number of sampled points
    time = np.linspace(0,T,N)
    field = np.sin(2.*np.pi*nupump*time)*np.exp(-0.5*((time-t0)/width)**2)
    freqs = np.fft.fftfreq(N,d=dt)
    energies = hplanck*freqs[0:int(N/2)]
    field_fourier = np.fft.fft(field)
    field_fourier = field_fourier[0:int(N/2)]
    field_int = field_fourier.real**2+field_fourier.imag**2
    field_int = field_int/max(field_int)
    position = np.where(energies>=trans_energy)[0][0]
    scale = field_int[position]
    if verbose:
        print('energy resolution in meV',1e3*(energies[1]-energies[0]))
        print('maximum energy',energies[-1])
        print('number of points of the FT',N)
        print('match with the trans_energy at',energies[position])
        print('scale',scale)
    return scale

##########################################################################################


