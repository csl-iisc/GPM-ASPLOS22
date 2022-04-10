# GPM: <ins>G</ins>PU with <ins>P</ins>ersistent <ins>M</ins>emory 
[![DOI](https://zenodo.org/badge/433300625.svg)](https://zenodo.org/badge/latestdoi/433300625)

GPM is a system which allows a GPU to leverage Persistent Memory and enables writing highly performant recoverable GPU applications.
The repository contains the source of our benchmark suite: GPMBench and a CUDA library: LibGPM. 
GPMBench comprises of 9 benchmarks categorized as transactional, native and checkpointing. 
LibGPM contains the source of our CUDA library which provides a user-friendly interface for GPU-accelerated recoverable applications. 


For full details refer to our paper:
- Shweta Pandey, Aditya K Kamath, and Arkaprava Basu. 2022. **GPM: Leveraging Persistent Memory from a GPU.** In _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '22)._ DOI:https://doi.org/10.1145/3503222.3507758 [[Paper]](https://www.csa.iisc.ac.in/~arkapravab/papers/ASPLOS22_GPM.pdf) [[Video]](https://www.youtube.com/watch?v=WER5mZPYFSc)

## Setting up PMEM [~10 minutes]
This section explains how to setup your NVDIMM config to be run in app direct mode. This also makes sure that all the PMEM strips are interleaved to attain maximum bandwidth. 
1. Install all the dependencies to support PMEM
`chmod +x dependencies.sh`
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
Further details on how to replicate results can be found in the [README](/GPMBench/README.md) in the GPMBench folder.

## Source code
The relevant source code for libGPM can be found in "[LibGPM](/LibGPM/)".
Further details on the source code and full API documentation can be found in the [README](/LibGPM/README.md) in the LibGPM folder.
