"""Robert
This is intended to be the main user interface module to be run
for the purposes of classifying and characterising nanostructures.
In theory, all of the sub-modules for this programme should be
found in the same directory and it should not matter from where
it is run. This is because the user will define the absolute path 
to their raw data and even the file names (and possible relative path).
"""




import time
import matplotlib.pyplot as plt
import numpy as np
import math

import pickle
import sys
import os
import wikiquote

#The following imports are the dependencies 
#found in the parent directory.

import Movie_Read as Read
from ase.io import read
import Kernels
from Kernels import KB_Dist as KB
import Distances as Dist
import Utility
#import CNA

def explode_dependencies(tu: tuple, System: dict, Settings: dict, Metadata: dict):
    quantity, params = tu
    d_unknown = quantity.dependencies
    dependencies = d_unknown if type(d_unknown) is list else d_unknown(
        System, params, Settings, Metadata)
    dependencies = [Utility.tuplize(d) for d in dependencies]
    dependencies = [dd for d in dependencies for dd in explode_dependencies(d)]
    return dependencies + [tu]

def prog_iter(system: dict = None, quantities: list = None, 
              settings: dict = None, start: int = 0, end: int = None,
              step: int = None, n_frames_extract: int = None):

    print("Welcome to this LoDiS post-processing scheme."
          "This script takes energy.out and movie.xyz files as arguments "
          "unless otherwise specified by name in the following input "
          "requests.")

    #Below is the general scheme by which one takes the trajectory input.

    start_time=time.time()
    print("Initialising" + wikiquote.quotes(wikiquote.random_titles(max_titles=1))[0])
    
    System = {**Utility.default_system, **({} if System is None else System)}
    Settings = {**Utility.default_settings, **({} if Settings is None else Settings)}
    Metadata = {}
    
    FileDirectory = System['base_dir']
    MovieFileLocation = FileDirectory + System['movie_file_name']
    EnergyFileLocation = FileDirectory + System['energy_file_name']
    
    MovieFile=read(MovieFileLocation, index = ':')
    all_positions = [atoms.get_positions() for atoms in MovieFile]
    all_atoms = [atoms.get_chemical_symbols() for atoms in MovieFile]
    
    Species = set(all_atoms)
    NFrames = len(all_positions)
    NAtoms = len(all_atoms)
    
    del(MovieFile,all_positions,all_atoms)
    
    Metadata['Species'] = Species
    Metadata['NFrames'] = NFrames
    Metadata['NAtoms'] = NAtoms
    
    print("This trajectory contains ", NAtoms, " atoms of type(s) ", Species, " over ", NFrames, " frames.")
    
    # Checking input arguments
    if step is not None and n_frames_extract is not None:
        raise ValueError('step and n_frames_extract cannot be specified simultaneously')
    if not (0 <= start <= NFrames):
        raise ValueError('start is out of range')
    if end is None:
        end = NFrames
    if not (start <= end <= NFrames):
        raise ValueError('end is out of range')
    n_frames_in_range = end - start
    if step is None and n_frames_extract is None:
        step = 1
    if step is None:
        # Finding the maximum step that spreads out in the range evenly
        step = math.floor(n_frames_in_range / n_frames_extract)
        end = start + n_frames_extract * step
    range_frames = range(start, end, step)
    if n_frames_extract is not None:
        assert len(range_frames) == n_frames_extract
    n_frames_extract = len(range_frames)
    print()
    print('Analyzing from frame %d (t = %.2E) to frame %d (t = %.2E) '
          'with step of %d frame(s) (%.2E), in total %d frames' %
          (start, start * step, end, end * step, step,
           step * step, n_frames_extract))
    
    # Checking and sanitizing input
    if quantities is None:
        quantities = []
    quantities = [Utility.input2key(quantity_input) for quantity_input in quantities]

    quantities_dep = [d for q in quantities for d in explode_dependencies(q)]

    quantities_dep = Utility.ordered_set(quantities_dep)
    
    # Inflate classes
    _ = []
    for q, params in quantities_dep:
        _.append(q(params, System, Settings, Metadata))
    key2quantities = dict(zip(quantities_dep, _))
    quantities_dep = _
    quantities = [key2quantities[q] for q in quantities]
    quantities2key = {v: k for (k, v) in key2quantities.items()}
    print("Calculating/Extracting the following quantities:")
    print([q.display_name for q in quantities])
    quantities_only_dep = [q for q in quantities_dep if q not in quantities]
    if len(quantities_only_dep) != 0:
        print("With the following intermediates")
        print([q.display_name for q in quantities_only_dep])
    
    formats = []
    for q in quantities:
        *dimen, dtype = q.get_dimensions(NAtoms, len(Species))
        if dimen == [1]:  # scalar data
            dimen = n_frames_extract
        else:
            dimen = n_frames_extract, *dimen
        formats.append((Utility.key2input(quantities2key[q]), dimen, dtype))
    yield Metadata, formats

    print()
    print("Let's do this! Time to rock and roll...")
    
    results = {}
    results['Metadata'] = Metadata
    
    for i_frame in range_frames:
        results_cache = {}
        for quantity in quantities_dep:
            if __debug__:
                quantity_start = time.time()
            result = quantity.calculate(i_frame, results_cache, Metadata)
            key = quantities2key[quantity]
            quantity_delta = time.time() - quantity_start
            print('frame', i_frame, key, 'took',
                '%.2f ms' % (quantity_delta.total_seconds() * 1000))
            results_cache[key] = result
            if key[1] is None:
                results_cache[key[0]] = result  # For dependency accessor

        for quantity in quantities:
            key = quantities2key[quantity]
            results[Utility.key2input(key)] = results_cache[key]
        yield results
    print('Got the data, boss!')

    for quantity in quantities_dep:
        quantity.cleanup()

    program_end = time.time()
    print('Post proccessing finished, took ',
          (program_end - start_time), ' seconds.')
    

def process(system: dict = None, quantities: list = None, settings: dict = None,
            start: int = 0, end: int = None, step: int = None, n_frames_extract: int = None, pbar=True):
    results = {}
    i = 0
    first = True
    for frame in process_iter(system, quantities, settings,
                              start, end, step, n_frames_extract, pbar, 'full'):
        if first:
            metadata, formats = frame
            results['metadata'] = metadata
            for key, dimen, dtype in formats:
                results[key] = np.empty(dimen, dtype)
            first = False
            continue
        for k, v in frame.items():
            results[k][i] = v
        i += 1

    return results



#A little bit of proof-of-concept code to plot the first
#few PDDFs using this scheme.

"""

if __name__ == '__main__':

    Positions=[];Distances=[];PDDF=[]

    for x in range(5):
        Positions.append(np.column_stack((Elements[x],Trajectory[x])))
        Distances.append(Dist.Distance.Euc_Dist(Positions[x]))   #All possible pairings
        PDDF.append(Kernels.Kernels.KDE_Uniform(Distances[x],0.25))
        plt.plot(PDDF[x][0],PDDF[x][1])
        plt.show()
        
"""