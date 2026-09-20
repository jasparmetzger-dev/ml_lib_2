# ml_lib_2

## Branch 2-ml-abcs

Implementing base classes for model (Layer, Container, Optimizer)

## Idea
The ml_lib_2 project is supposed to be an implementation of classical machine learning methods based only on the standard library. This will of course be significantly slower than using *numpy*, but it teaches concepts of ND-shaping. It also gives this project it´s own math library.

## Structure

### Source code
The source code can be found in *src/*.

In *src/* there is a math library (*src/math/*) with the main part being the tensor class (*src/math/tensor.py*), which the models will use to store and train their weights and biases.

In *src/model/*, layer types, optimizers, and containers will be defined abstractly and then in concrete classes.

In *src/data/* data will be prepped to be fed into a ml-algorithm.

### Testing

All testing can be found in *tests/*. Mainly unit tests will be used. For this, *tests/unit/* will mock the structure of *src/* and test each file.
Integration tests are also possible and will mainly be used to test the integration of different *src/model/* modules and how they work together.
