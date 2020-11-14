from mppi import Utilities as U, Parsers as P
import numpy as np

def eval_transition_dipole(ndb,kpoint=0,transition=[0,1],component=0):
    """
    Compute the transition dipole of a specific transition
    
    Args:
        ndb : name of the dipoles database
        kpoint : selected kpoint
        transition : transitions activated by the pump
        component : cartesian component of the dipole
    """
    dipoles = U.get_variable_from_db(ndb,'DIP_iR')
    dip = dipoles[kpoint][transition[0]][transition[1]][component]
    dip_mod = np.sqrt(dip[0]**2+dip[1]**2)
    return dip_mod

def eval_trans_energy(save,kpoint=0,transition=[0,1],
                      set_scissor=None,set_gap=None,set_direct_gap=None):
    """
    Compute the energy of the selected transition
    
    Args:
        save : SAVE folder that contain the ns.db1 database
        kpoint : selected kpoint
        transition : transitions activated by the pump
        component : cartesian component of the dipole
    """
    data = P.NsdbsParser(save=save,verbose=False)
    trans_energy = data.get_transitions(initial=[transition[0]],final=[transition[1]],
                    set_scissor=set_scissor,set_gap=set_gap,set_direct_gap=set_direct_gap)
    return trans_energy[kpoint][0]

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

def eval_pulse_area_single_trans(dip_mod,trans_energy,field_int,width,pump_energy,verbose=False):
    """
    Compute the coupling frequency and the pulse area in function of the field intensity,
    for a single k-point and a single transition
    
    Args:
        dip_mod : value of the dipole for the selected transition
        trans_energy : energy of the transition
        field_int : field intensity in kW/cm^2
        width : the width of the field in fs
        kpoint : selected kpoint
        transition : transitions activated by the pump
        component : cartesian component of the dipole
    """
    Z0 = U.vacuum_impedence
    eff_field_int = field_int*eval_effective_field_int(width,pump_energy,trans_energy)
    eff_field_int = eff_field_int*1e3*1e4 #W/m^2
    field_amp = np.sqrt(Z0*eff_field_int) #V/m
    field_amp = field_amp*U.Bohr_radius #V/a0 in atomic units
    coupling_frequency = dip_mod*field_amp*2*np.pi/U.Planck_ev_ps #THz
    
    theta = np.sqrt(2*np.pi)*width*coupling_frequency*1e-3
    if verbose:
        print('coupling frequency (THz):',coupling_frequency)
        print('pulse area :',theta)
    return coupling_frequency,theta








##########################################################################################


def eval_field_intensity(ndb,theta,width,kpoints=None,transitions=[[0,1]],component=0):
    """
    Compute the field intensity that produce the pulse are given as input.
    The transition dipoles are computed by averaging the dipole over the kpoints provided
    in the kpoints field. If the field is None all the kpoints of the ndb database are
    considered. The function sums over all the transitions included in the
    transitions field.
    
    Args:
        ndb : name of the dipoles database
        theta : pulse area
        width : the width of the field in fs
        kpoints : list with the selected kpoint. If None all the kpoints of the database are included
        transitions : list with the transitions activated by the pump
        component : cartesian component of the dipole
    """
    Z0 = U.vacuum_impedence
    dip_mod = eval_transition_dipole(ndb,kpoints=kpoints,transitions=transitions,component=component)
    
    coupling_frequency = theta/(np.sqrt(2*np.pi)*width*1e-3) #THz
    field_amp = coupling_frequency/(dip_mod*2*np.pi/U.Planck_ev_ps) #V/a0 in atomic units
    field_amp = field_amp/U.Bohr_radius #V/m
    field_int = field_amp**2/Z0 #W/m^2
    field_int = field_int*1e-3*1e-4 #kW/cm^2
    print('field intensity (kW/cm^2) :',field_int)
    return field_int