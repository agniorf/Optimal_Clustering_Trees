[Data]
                                  PruneSet = 0.25
                                  TestSet = None
                                  
                                  [General]
                                  RandomSeed = 1

                                  [Constraints]
                                  MaxDepth = 1
                                  
                                  [Attributes]
                                  Target = 1-2
                                  Clustering = 1-2 
                                  Descriptive = 1-2 
                                  
                                  [Tree]
                                  Heuristic = VarianceReduction
                                  
                                  [Output]
                                  WritePredictions = {Train}
                                  