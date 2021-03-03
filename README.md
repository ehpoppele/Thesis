# Thesis
Senior Undergrad Thesis Investigating Neuroevolution

This readme is a work in progress

## Installation
pip install gym

pip install gym[atari] (requires cmake; apt install cmake)

install pytorch as desired, following instructions on page (pip install torch torchvision); cuda if you want to use gpu

## Possible Errors/Issues
If the program crashes with a "too many open files" error, trying raising the user file limit to 4096, or reducing the population size (for multithreading only).

