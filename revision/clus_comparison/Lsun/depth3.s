[Data]
PruneSet = 0.25
TestSet = None

[General]
RandomSeed = 1

[Model]
MinimalWeight = 5.0

[Constraints]
MaxDepth = 3

[Attributes]
Target = 1-2
Clustering = 1-2 
Descriptive = 1-2 

[Tree]
Heuristic = VarianceReduction

[Output]
WritePredictions = {Train}