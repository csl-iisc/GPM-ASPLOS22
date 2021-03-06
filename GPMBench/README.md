# GPMBench
GPMBench comprises of 9 benchmarks categorized as transactional, native and checkpointing. 
The artifact also allows a user to reproduce some of the key results published in the paper.
These key results include: 
1. Figure 1: Benefits of GPM over CPU with PM. 
2. Figure 9: Benefits of GPM over other methods of persistence.
3. Table 5: Restoration latency (RL) in GPM. 
2. Figure 10: Implications of eADR on GPM.
2. Figure 11a: Benefit of HCL.

## Steps to setup and replicate results
The following steps are required to reproduce the results, along with the expected run time. All commands should be run in this folder.
 1. **Setting up PMEM [~10 minutes]**
 2. **Setting up cuDNN [~15 minutes]**
 3. **Replicating Figure 1 [~40 + 30 minutes]**
 4. **Replicating Figure 9 [~120 minutes]**
 5. **Replicating Table 5 [~7 minutes]**
 6. **Replicating Figure 10 [~120 minutes]**
 7. **Replicating Figure 11a [~5 minutes]**


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

**Figure 1 [~40 + 30 minutes]**    
Run the following command:
```
make figure_1
```
This will run the appropriate benchmarks and measure the run time.    

Raw outputs and run times will be contained in the results folder in Figure1/Figure1a and Figure1/Figure1b.

To obtain the report, run the following command: 
```
make out_figure1
```

Final normalized results will be outputted in the terminal and are also contained in this folder in tab-separated format. This can be imported into a spreadsheet of your choice to generate the appropriate figure.


**Figure 9 [~120 minutes]**     
Run the following command:
```
make figure_9
```
This will run the benchmarks for GPM and CAP and measure the run time. 

Raw outputs will be kept in individual results folders that can be obtained in GPMbench/transactional, GPMBench/checkpointing and GPMBench/native.

To obtain the report, run the following command: 
```
make out_figure9
```

Final normalized results will be outputted in the terminal and are also contained in this folder in tab-separated format. This can be imported into a spreadsheet of your choice to generate the appropriate figure.


**Table 5 [~7 minutes]**     
Run the following command:
```
make table_5
```
This will run the crash-recovery kernels for GPM benchmarks and measure their time. 

Raw outputs will be kept in results folder in GPMbench/transactional, GPMBench/checkpointing and GPMBench/native.

To obtain the report, run the following command: 
```
make out_table5
```

Final normalized results will be outputted in the terminal and are also contained in this folder in tab-separated format.


**Figure 10 [~120 minutes]**     
Run the following command:
```
make figure_10
```
This will run the benchmarks for GPM and CAP and measure the run time. 

Raw outputs will be kept in individual results folders that can be obtained in GPMbench/transactional, GPMBench/checkpointing and GPMBench/native.

To obtain the report, run the following command: 
```
make out_figure10
```

Final normalized results will be outputted in the terminal and are also contained in this folder in tab-separated format. This can be imported into a spreadsheet of your choice to generate the appropriate figure.

**Figure 11a [~5 minutes]**     
Run the following command:
```
make figure_11a
```
This will run the benchmarks for GPM with and without HCL and measure the run time. 

Raw outputs will be kept in individual results folders that can be obtained in GPMbench/transactional, GPMBench/checkpointing and GPMBench/native.

To obtain the report, run the following command: 
```
make out_figure11a
```

Final normalized results will be outputted in the terminal and are also contained in this folder in tab-separated format. This can be imported into a spreadsheet of your choice to generate the appropriate figure.
