import unittest
import ising
import copy
import random
import lib490.extraassert as extraassert

class IsingTest(unittest.TestCase,extraassert.ExtraAssert):

    def test_quench(self):
        '''
        Performs a sequence of known downhill moves until we get to the first rejected uphill move.
        '''
        prng = random.Random(1)

        # expected random number sequence for this seed
        # all moves are downhill or dEnergy = 0 except the last
        index_sequence = [4,2,8,3,15,14,15,12,6,3,15]
        
        # by using init_value=1, no random numbers are pulled. This is
        # important to obtain the expected index_sequence
        obj = ising.Ising([4,4],temperature=.0001,
                          init_value=1,prng=prng)

        # set the initial grid
        obj.grid=[[-1., 1.,-1., 1.],
                  [ 1.,-1., 1.,-1.],
                  [-1., 1.,-1., 1.],
                  [ 1.,-1., 1.,-1.]]

        # expected state after the sequence of moves
        final_grid=[[-1., 1., 1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [ 1., 1.,-1., 1.],
                    [-1.,-1.,-1.,-1.]]

        # iterate through the sequence
        obj.iterate(len(index_sequence))

        with self.subTest(test="Final grid"):
            self.assertMatrixAlmostEqual(final_grid,obj.grid)
        with self.subTest(test="Final energy"):
            self.assertEqual(-4,obj.calc_energy_magentization()[0])

            
    def test_calc_energy_magentization(self):
        '''
        Test energy and magnetization calculation
        '''
        obj = ising.Ising([4,4])
        obj.grid=[[-1., 1.,-1., 1.],
                  [ 1.,-1., 1.,-1.],
                  [-1., 1.,-1., 1.],
                  [ 1.,-1., 1.,-1.]]
        with self.subTest(quantity="Energy"):
            self.assertAlmostEqual(obj.calc_energy_magentization()[0],32.)
        with self.subTest(quantity="Magnetization"):
            self.assertAlmostEqual(obj.calc_energy_magentization()[1],0.)
           
    def test_move(self):
        '''
        Test energy change, move and no move
        '''
        obj = ising.Ising([4,4])
        obj.grid=[[-1., 1.,-1., 1.],
                  [ 1.,-1., 1.,-1.],
                  [-1., 1.,-1., 1.],
                  [ 1.,-1., 1.,-1.]]
        original_grid=copy.deepcopy(obj.grid)
        
        with self.subTest(quantity="dEnergy"):
            self.assertAlmostEqual(obj.move(0, False), -4.0) 
        
        with self.subTest(quantity="No move"):
            obj.move(15, False)
            self.assertMatrixAlmostEqual(original_grid, obj.grid)
        
        with self.subTest(quantity="Move"):
            obj.move(15, True)
            original_grid[3][3] = -original_grid[3][3]
            self.assertMatrixAlmostEqual(original_grid, obj.grid)
        
if __name__ == "__main__":
    unittest.main()
