apt-get update

#PMEM dependencies

apt-get install -y libpmem1 librpmem1 libpmemblk1 libpmemlog1 libpmemobj1 libpmempool1
apt-get install -y libpmem-dev librpmem-dev libpmemblk-dev libpmemlog-dev libpmemobj-dev libpmempool-dev libpmempool-dev
#sudo apt-get install libpmem1-debug librpmem1-debug libpmemblk1-debug libpmemlog1-debug libpmemobj1-debug libpmempool1-debug

#Intel PM CTL installation
apt-get install -y ipmctl

#Linux NVDIMM CTL installation
apt-get install -y git gcc g++ autoconf automake asciidoc asciidoctor bash-completion xmlto libtool pkg-config libglib2.0-0 libglib2.0-dev libfabric1 libfabric-dev doxygen graphviz pandoc libncurses5 libkmod2 libkmod-dev libudev-dev uuid-dev libjson-c-dev libkeyutils-dev
apt-get install ndctl

#CuDNN installation
apt-get install -y zlib1g

#For the CPU workloads
apt-get install -y gcc libomp-dev libpthread-stubs0-dev build-essential aptitude libstdc++6
