# Visual Cortex Modelling

This code is a simplified version of my Master's Thesis at the University of Cambridge. In a nutshell, the project involves modelling the primary visual using an RNN architecture with paramters which depends on the connectivity type (excitatory and inhibitory). With these connectivities, we then execute the network and perform backpropagation to find the parameters which generate a network which produces tuning curves that are most closely related to actual mouse V1 responses. To understand the project further, please refer to the final report attached in this repository. 

As this repo is aimmed towards scientists to model the V1 as an RNN on GPU, some aspect of the reports were removed. This include the usage of the xNES algorithm and CNN architecture. If you would like to see the code or have any questions please contact me at https://www.linkedin.com/in/chulabutra-chuenchoksan/.

To navigate the repository:

1. The feed-forward and RNN weights generation and network execution classes are in the `rat` folder.
2. As most files will import from `rat`, files which are in folders will likely not work unless `rat` is installed into an env or `rat` is moved into the folder.
3. `utils` contains useful functions which are repetitive across experiments and scripts. This include a forward differentiator which is not fully implemented on PyTorch.
4. `rodents_plotter` will plot generic graphs and plots for quick analysis of parameter values.
