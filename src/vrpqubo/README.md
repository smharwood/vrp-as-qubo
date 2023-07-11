# Generating a test set

Run 
```
python generate_test_set.py -p TestSet -t 20 30 40 50
```
to generate a test set of various sized problems in the folder `TestSet`.  
This will include "feasibility" versions (minimum objective value equals zero) of each optimization problem as well.

Note: I can only guarantee that the `mirp_g1.py` examples work -- others may require a little extra effort.