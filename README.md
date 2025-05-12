# DAC of Feed Pump using multi-stage NMPC

This repository contains simulations for degradation-aware control (DAC) of the feed pump using the multi-stage NMPC algorithm, supporting our submission to the IEEE International Conference on Instrumentation, Control, and Automation (ICA).

## Purpose

Sharing MATLAB code and data for reproducing results and facilitating review.

## Repository Structure

* **`TestRegime.py`**: Python simulation code for optimization of proof test regime.
* **`CompPFD.py`**: Python simulation code to compare varying and constant interval of proof test regimes.
* **`MarkovLib.py`**: Library of functions to accompany `TestRegime.py` and `CompPFD.py` for realization of degradation model.
* **`FeedMPC.m`**: MATLAB simulation code for DAC of the feed pump using multi-stage NMPC.

## Requirement

* [Matlab Optimization Toolbox](https://se.mathworks.com/products/optimization.html)
* [YALMIP: A Toolbox for Modeling and Optimization in MATLAB](https://yalmip.github.io)

## IEEE International Conference on Instrumentation, Control, and Automation Submission

This repository directly supports data and code for our paper submission.
