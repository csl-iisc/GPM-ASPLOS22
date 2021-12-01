# GPM: <ins>G</ins>PU with <ins>P</ins>ersistent <ins>M</ins>emory 
GPM is a system which allows a GPU to leverage Persistent Memory and enables writing highly performant recoverable GPU applications.
The repository contains the source of our benchmark suite: GPMBench and a CUDA library: LibGPM. 
GPMBench comprises of 9 benchmarks categorized as transactional, native and checkpointing. 
LibGPM contains the source of our CUDA library which provides a user-friendly interface for GPU-accelerated recoverable applications. 
More details about the work can be found in our paper ASPLOS'22 paper: Leveraging Persistent Memory from a GPU. 
The artifact also allows a user to reproduce some of the key results published in the paper.
These key results include: 
1. Figure 1: Benefits of GPM over CPU with PM. 
2. Figure 9: Benefits ofGPMover CPU with PM.
3. Table 5: Restoration latency (RL) in GPM. 


## Steps to setup and replicate results
The following steps are required to reproduce the results, along with the expected run time. All commands should be run in the main repository folder.
 1. **Setting up PMEM [~10 minutes]**
 2. **Setting up cuDNN [~15 minutes]**
 3. **Replicating Figure 1 [~40 + 30 minutes]**
 4. **Replicating Figure 9 [~120 minutes]**
 5. **Replicating Table 5 [~7 minutes]**


## Setting up PMEM [~10 minutes]
This section explains how to setup your NVDIMM config to be run in app direct mode. This also makes sure that all the PMEM strips are interleaved to attain maximum bandwidth. 
1. Install all the dependencies to support PMEM
`sudo ./dependencies.sh`
2. Run the teardown script to tear down any older PMEM configuration. 
`sudo ./pmem-setup/teardown.bashrc`
3. Run the preboot script to destroy all the existing namespaces. This script will also reboot the sytsem. 
`sudo ./pmem-setup/preboot.bashrc`
4. Run the config-setup script to configure interleaved namespace for PMEM along with app-direct mode. To run the script one has to be root. 
```
sudo su 
./pmem-setup/config-setup.bashrc
exit
```

## Setting up cuDNN [~15 minutes]
Download CuDNN 8.2 for CUDA 11.X from Nvidia's website.
One can find the CuDNN 8.2 libraries at: https://developer.nvidia.com/rdp/cudnn-archive
Follow the instructions from: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_800_ea/cudnn-install/index.html#download to complete installation. 

More specifically - 

1. Choose a path for installing CuDNN: \<cudnnpath\>
2. Install the downloaded runtime library as:
sudo dpkg -i libcudnn8_x.x.x.x-1+cudax.x_amd64.deb
3. Install the developer library, for example:
sudo dpkg -i libcudnn8-dev_8.x.x.x.x-1+cudax.x_amd64.deb
4. Install the code samples and the cuDNN library documentation, for example:
sudo dpkg -i libcudnn8-doc_8.x.x.x.x-1+cudax.x_amd64.deb


## Replicating primary results (Figures and Tables)
We provide the scripts required to compile and generate the results contained in the paper.

**Figure 1 [~40 + 30 minutes]**    
Run the following command in the main repository folder:
```
make figure_1
```
This will run the appropriate benchmarks and measure the run time.    

Raw outputs and run times will be contained in the results folder in Figure1/Figure1a and Figure1/Figure1b.

Final normalized results will be outputted in the terminal and are also contained in the reports/ folder as a tab-separated format. This can be imported into a spreadsheet of your choice to generate the appropriate figure.


**Figure 9 [~70 minutes]**     
Run the following command in the main repository folder:
```
make figure_9
```
This will run the benchmarks for GPM and CAP and measure the run time. 

Raw outputs will be kept in individual results folders that can be obtained in GPMbench_LibGPM/transactional, GPMBench_LibGPM/checkpointing and GPMBench_LibGPM/native.

Final normalized results will be outputted in the terminal and are also contained at reports/ in tab-separated format. This can be imported into a spreadsheet of your choice to generate the appropriate figure.


**Table 5 [~7 minutes]**     
Run the following command in the main repository folder:
```
make table_5
```
This will run the crash-recovery kernels for GPM benchmarks and measure their time. 

Raw outputs will be kept in results folder in GPMbench_LibGPM/transactional, GPMBench_LibGPM/checkpointing and GPMBench_LibGPM/native.

Final normalized results will be outputted in the terminal and are also contained at reports/ in tab-separated format.

## Source code
The relevant source code for libGPM can be found in "[GPMBench_LibGPM/libgpm/include](/GPMBench_LibGPM/libgpm/include)".
This folder contains 7 files, which we explain below:
* libgpm.cuh - Contains the main implementation details of GPM, which allow for allocation/deallocation on PMEM.
* libgpmlog.cuh - Contains the relevant implementation of logging (HCL and conventional) in GPM.
* libgpmcp.cuh - Contains the relevant implementation of checkpointing in GPM.
* bandwidth_analysis.cuh - Contains helper definitions used for bandwidth measurement.
* gpm-annotations.cuh - Contains helper functions used to calculate write amplification.
* change-ddio.h - Contains functions to turn DDIO off and on. [cite]
* gpm-helper.cuh - 

