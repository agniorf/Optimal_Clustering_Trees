[Data]
                                  PruneSet = 0.25
                                  TestSet = None
                                  
                                  [General]
                                  RandomSeed = 5

                                  [Constraints]
                                  MaxDepth = 1
                                  
                                  [Attributes]
                                  Target = 1-3
                                  Clustering = 1-3 
                                  Descriptive = 1-3 
                                  
                                  [Tree]
                                  Heuristic = VarianceReduction
                                  
                                  [Output]
                                  WritePredictions = {Train}
                                  