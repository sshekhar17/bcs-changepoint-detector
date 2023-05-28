# Sequential Changepoint Detection via Backward Confidence Sequences 

**Summary:** we present a general framework for constructing sequential changepoint detection methods using confidence sequences. This repository contains the following instantiations of the general scheme: 
1. Detecting mean-shift in univariate Gaussian observations (`Experiment_GaussianMean.py`) 
2. Detection mean-shift in univariate bounded observations  (`Experiment_BoundedMean.py`)
3. Detection changes in CDFs (`Experiment_CDF.py`)
4. Detecting changes in the distribution of paired observations (`Experiment_TwoSample.py`)
5. Detecting harmful distribution shifts in binary classification(`Experiment_BinaryClassifier.py`) 
