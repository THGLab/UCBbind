8/4/2025
----------

This iteration of UCBBind leverages ligand similarity when a joint protein-ligand pair is not found in the reference set. 

Module Y is thus composed of two components: Y-joint and Y-ligand-only. 

Y-joint operates the same way Module Y did in previous iterations of UCBBind.
Y-ligand-only is a fallback mechanism in the case that:
i) there is a similar ligand to the query ligand
ii) Y-joint was not used

Y-ligand-only takes the average of the experimentally measured binding affinities of the most similar ligand to the query ligand.

It then adds a predicted residual to the average to yield the final binding affinity prediction.

Note that this ridge regression model takes in different input features than the one that predicts the residual of the Y-joint pairs.

These features include:
- Binding Count: The number of binding affinity measurements in the reference dataset (e.g., BindingDB) for the most similar ligand across its known protein targets.
- Mean affinity: The average experimentally measured binding affinity across these proteins.
- Minimum affinity: The lowest measured binding affinity observed.
- Maximum affinity: The highest measured binding affinity observed.
- Dominance ratio: The fraction of the total ligand similarity contributed by the most similar ligand 
