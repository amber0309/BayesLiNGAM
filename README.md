# BayesLiNGAM

Python code of causal discovery algorithm for causal graphs proposed in

[Bayesian discovery of linear acyclic causal models](https://arxiv.org/abs/1205.2641)  
Hoyer, Patrik O., and Antti Hyttinen.  
Conference on Uncertainty in Artificial Intelligence (**UAI**) 2009.

## Prerequisites

- numpy
- scipy
- sklearn
- itertools
- copy

We test the code using Anaconda 4.3.30 64-bit for python 2.7 on Windows 10. Any later version should still work perfectly.

## Running the test

After installing all required packages, you can run *demo.py* to see whether **BayesLiNGAM** could work normally.

The test code does the following:

1. it generates 1000 observations (a (1000, 2) *numpy array*) from a causal model with 2 variables;
2. BayesLiNGAM is applied on the generated data to infer the true causal graph.

## Apply **BayesLiNGAM** on your data

### Usage

```python
mdl = BayesLiNGAM(X, B)
mdl.inference()
```

### Description

Class `BayesLiNGAM()`

| Argument  | Description  |
|---|---|
|X | matrix of all instances, (n_samples, n_vars) numpy array |
|B | true causal graph skeleton, (n_vars, n_vars) numpy array |

After the initialization of `BayesLiNGAM()` object, use function `self.inference()` to estimate the causal graph.

## Author

- **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/Multidomain-Discriminant-Analysis/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- We appreciate the [CIDG](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16595) [code](https://mingming-gong.github.io/papers/CIDG.zip) by Ya Li.
