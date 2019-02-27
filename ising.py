import random
import metropolis as metropolis
import csv
import math

class Ising(object):

    def __init__(self,gridsize,kb=1,temperature=0,interaction_const=1,
                 init_value=None,
                 thermo_filename=None,
                 grid_filename=None,
                 write_steps=1,thermo_steps=1,
                 prng=random):
        '''
        Create a new Ising grid.
        Args:
            gridsize: (list of two ints) x and y grid dimensions
            kb: (float) Boltzmann's constant in desired units
            temperature : (float) temperature in desired units
            interaction_const: (float) interaction constant in Ising
                model
            initValue: (-1,1, or None) assign all spins to this state or random if None
            
            thermo_filename: (str) file to write per step CSV
                thermodynamic data to
            grid_filename: (str) file to write per step CSV grid data to
            write_steps : (int) number of steps between write coordinates 
            thermo_steps : (int) number of steps between update thermodynamic statistics
            prng: (Random) pseudo random number generator that has
                random(), choice() and randint() methods

        Side Effects:
           creates supplied file names
        '''
        self.gridsize = gridsize
        self.kb=kb
        self.temperature = temperature
        self.interaction_const = interaction_const

        self._set_csv_writer("thermo",thermo_filename)
        if self.thermo_filehandle:
            self.thermo_filehandle.write(
                '"'+'","'.join(["Step","Energy","Magnetization"])+'"\n')
            
        self._set_csv_writer("grid",grid_filename)
        
        self.write_steps=write_steps # write every nSteps... integer
        self.thermo_steps=thermo_steps
        
        self.prng=prng

        self.gridCount = [[0]*self.gridsize[1] for i in range(self.gridsize[0])]
        if init_value is None:
            self._init_grid_random()
        else:
            self._init_grid_value(init_value)
        if self.grid_csv:
            self.grid_csv.writerow(self.gridsize)
        self.calc_energy_magentization()

        self.reset_stats()

    def _init_grid_random(self):
        '''
        Assign random spin states to all grid points
        '''
        self.grid = [[0]*self.gridsize[1] for i in range(self.gridsize[0])]
        for irow in range(self.gridsize[0]):
            for icol in range(self.gridsize[1]):
                self.grid[irow][icol] = self.prng.choice([-1,1])

    def _init_grid_value(self,value):
        '''
        Assign a single spin state to all grid points
        Args:
            value: (int) 1 or -1.
        '''
        self.grid = [[0]*self.gridsize[1] for i in range(self.gridsize[0])]
        for irow in range(self.gridsize[0]):
            for icol in range(self.gridsize[1]):
                self.grid[irow][icol] = value
        self.grid[0][2] = -value

    def _set_csv_writer(self, filetype, filename):
        '''Stores filenname, filehandle and csv writer as attributes.  Creates
        attributes filetype starting with <filetype> and ending with
        "_filename", "_filehandle" and "_csv". If filename is None,
        all attributes have value None.
        Args:
            filetype: (str) type of file 
            filename: (str) file name to write to
        SideEffect:
            opens filename for writting.  Modifies attributes.
        '''
        setattr(self,filetype+"_filename", filename)
        setattr(self,filetype+"_filehandle",None)
        setattr(self,filetype+"_csv",None)
        if filename:
            setattr(self,filetype+"_filehandle",
                    open(filename,"w",newline='\n'))
            setattr(self,filetype+"_csv",
                    csv.writer(getattr(self,filetype+"_filehandle")))

    def iterate(self,nsteps):
        '''
        Iterate nsteps using the metropolis method.
        Args:
            nsteps: (int) number of trial moves to attempt
            write_steps: (int) save to defined files and update
                statistics every write_steps.
        Side Effects:
            Updates writes to file, updates self.grid
        '''
        metropolis.iterate(nsteps,self.temperature,
                           self.gridsize[0]*self.gridsize[1],
                           self.move,self.process,self.prng)
                            
    def reset_stats(self):
        '''
        Reset energy and magnetization statistics.
        '''
        self.nsamples=0
        self.energy_ave=0.
        self.energy_var_recurrance=0.
        self.magnetization_ave=0.
        self.magnetization_var_recurrance=0.

    def update_stats(self):
        '''
        Update running average and variance for energy and magnetization
        '''
        self.nsamples+=1
        #energy
        delta = self.energy - self.energy_ave
        self.energy_ave += delta/self.nsamples
        self.energy_var_recurrance += delta*(self.energy - self.energy_ave)

        #magnetization
        delta = self.magnetization - self.magnetization_ave
        self.magnetization_ave += delta/self.nsamples
        self.magnetization_var_recurrance += delta*(self.magnetization - self.magnetization_ave)


    def get_energy_per_spin(self):
        '''
        Return the mean, variance and standard error in the mean (SEM)
        of the energy per spin over the simulation
        '''
        ave = self.energy_ave/self.gridsize[0]/self.gridsize[1]
        var = (self.energy_var_recurrance/(self.nsamples-1)
               /self.gridsize[0]/self.gridsize[1])
        sem = math.sqrt(var/(self.nsamples-1))
        return ave, var, sem


    def get_specific_heat_per_spin(self):
        '''
        Return the specific heat per spin.
        '''
        return (self.get_energy_per_spin()[1]
                /self.kb/self.temperature**2)

    def get_magnetization_per_spin(self):
        '''
        Return the mean, variance and standard error in the mean (SEM)
        of the magnetization per spin over the simulation
        '''
        ave = self.magnetization_ave/self.gridsize[0]/self.gridsize[1]
        var = (self.magnetization_var_recurrance/(self.nsamples-1)
               /self.gridsize[0]/self.gridsize[1])
        sem = math.sqrt(var/(self.nsamples-1))
        return ave, var, sem

    def get_magnetic_susceptibility_per_spin(self):
        '''
        Return the magnetic susceptibility per spin
        '''
        return (self.get_magnetization_per_spin()[1]
                /self.kb/self.temperature)

    def calc_energy_magentization(self):
        '''Calculate the total energy and magentization of the grid
        with periodic boundary conditions.
        Returns:
           (float,float) the value of the total energy and magnetization
        Side Effects:
           Updates self.energy and self.magnetization
        '''
        self.energy = 0
        self.magnetization = 0
        for irow in range(self.gridsize[0]):
            for icol in range(self.gridsize[1]):
                
                self.energy += -self.interaction_const*(
                    self.grid[irow][icol]*self.grid[irow][(icol+1)%self.gridsize[1]]
                    +self.grid[irow][icol]*self.grid[(irow+1)%self.gridsize[0]][icol]
                    )
                
                self.magnetization += self.grid[irow][icol]
        return self.energy,self.magnetization
        
    def move(self,index,update):
        '''
        Performs a trial move if update is False to get 
        the energy difference.
        An actual move is made if update is True.
        
        If update is False, calculate the change in energy of 
        flipping the index spin and return this value
        but do not change the state of the spin. 
           
        If update is True, flip the state of the index spin and 
        update self.energy and self.magnetization. 

        Args:
            index: index number of the atom to move
            update: if 'False' generate the trial move and return the
                     energy difference (new-old). If 'True' sets the
                     new grid position.
    
        Returns:
            the energy difference if update == False, otherwise,
            returns nothing

        Side Effect:
            Grid position is changed if update is true
            updates self.magnetization and self.energy 
        '''
        irow = index//self.gridsize[0]
        icol = index%self.gridsize[0]

        dEnergy = self.interaction_const*(self.grid[irow][icol]*self.grid[irow][(icol+1)%self.gridsize[1]]
                                          +self.grid[irow][icol]*self.grid[(irow+1)%self.gridsize[0]][icol]
                                          +self.grid[irow][icol]*self.grid[irow][(icol-1)%self.gridsize[1]]
                                          +self.grid[irow][icol]*self.grid[(irow-1)%self.gridsize[0]][icol])

        if not update:
            return dEnergy

        if update:
            self.grid[irow][icol] = -self.grid[irow][icol] 
            self.magnetization += 2*self.grid[irow][icol]
            self.energy += dEnergy
        
           

    def process(self,step,*args,**kwargs):
        """
        Processes the current state for output if step is a multiple
        of writeSteps. Updates all thermodynamics statistics and
        writes out the current state and thermodynamics, if the file
        handles are set.
        
        Args:
            step: current iteration step
        
        Side Effects:
            thermodynamics statistics may be updated and files may be
            written to
        """
        if self.write_steps > 0 and step%self.write_steps == 0 :
            self.print_grid(step)
        if self.thermo_steps > 0 and step%self.thermo_steps == 0 :
            self.update_stats()
            self.print_thermo(step)

    def print_thermo(self,step):
        '''
        Write out current step, energy and magnetization.
        Args:
            step: (int) current iteration step
        Side Effects: 
            writes to file
        '''
        if self.thermo_csv:
            self.thermo_csv.writerow([step,self.energy,self.magnetization])

    def print_grid(self,step):
        '''
        Write out current grid values.
        Args:
            step: (int) current iteration step
        Side Effects: 
            writes to file
        '''
        if self.grid_csv:
            for row in self.grid:
                self.grid_csv.writerow(row)
        
