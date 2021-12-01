# bash script to configure the Optane persistent memory

# --------- Configuration Topology ---------------
# interleaved single pmem device
# created using Linux Device Mapper
# device: /dev/amapper/striped-pmem; mount point: /pmem


# CPU Socket 0 - AppDirect, interleaved, fsdax mode
# CPU Socket 1 - AppDirect, interleaved, fsdax mode
# CPU Socket 2 - AppDirect, interleaved, fsdax mode
# CPU Socket 3 - AppDirect, interleaved, fsdax mode
# ------------------------------------------------

ndctl destroy-namespace all --force

echo "y" | ipmctl create -goal

reboot
