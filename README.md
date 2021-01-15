# REVEALING BRAIN STATES WITH RESERVOIR COMPUTING - TFG - Arnau Naval Ruiz

_The purpose of the project is to build a Reservoir Computing model capable of identifying different movement related states from a neural data set recorded from a non-human primate. To this end, we will perform the following analyses:

*	Five-State Classification: where we will try to distinguish all five states at the same time, giving it a full trial to classify.

*	Left vs Right: where we will try to distinguish when the data is from the primate using the left arm or the right arm.

Not only it is interesting to see if RC is capable of capturing the features recorded from specific brain regions, we will also see if the representation of the data after going through the model helps classify the states into their belonging classes better than without using the neural network.


## Files description

* data.py : This file contains all the code regarding the data structures used and data processing

* network.py : This file contains the structure of the network and all the processes applied to it

* reservoir.py : This file runs the project, it builds the data and network using both the mentioned files above.

## How to run the code

_There are 2 tests to perform:

* Five-State Classification: that uses either Logistic Classifier (log) or Linear Classifier (lin). Command -> 5s

* Left vs Right: that uses either Logistic Classifier (log) or 1NN Classifier (1nn). Command -> lvr

Both tests can apply a range of frequency filters:

Name - command
*Baseline - baseline
*Theta - theta
*Alpha - alpha
*Beta - beta
*Low gamma - lowgamma
*High gamma - highgamma
*Low ripple - lowripple
*High ripple - highripple
*Low multi-unit - lowmultiunit

In order to run the console we will use the following command: 

```
python3 {test_command (string)} {filter_command (string)} {classifier_command (string)} {number of nodes (int)} {input probability connection (float)} {reservoir probability connection (float)}
```

An example for 5 state classification would be:
```
python3 5s lowmultiunit lin 100 0.5 0.2
```

An example for Left vs Right classification would be: 

```
python3 lvr theta 1nn 50 0.2 0.4
```

The following combinations are not supported:

* Using 1NN Classifier (1nn) with the 5 state problem

* Using Linear Classifier (lin) with the Left vs Right problem
