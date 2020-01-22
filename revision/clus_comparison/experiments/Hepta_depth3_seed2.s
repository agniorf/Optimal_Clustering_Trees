[Data]
                                  PruneSet = 0.25
                                  TestSet = None
                                  
                                  [General]
                                  RandomSeed = 2

                                  [Constraints]
                                  MaxDepth = 3
                                  
                                  [Attributes]
                                  Target = 1-3
                                  Clustering = 1-3 
                                  Descriptive = 1-3 
                                  
                                  [Tree]
                                  Heuristic = VarianceReduction
                                  
                                  [Output]
                                  WritePredictions = {Train}
                                  