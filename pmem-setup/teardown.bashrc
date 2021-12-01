# bash script to configure the Optane persistent memory

# script to remove the single pmem device configuration across all CPU sockets
# Configuration was created using Linux Device Mapper
# device: /dev/mapper/striped-pmem; mount point: /pmem

umount /dev/mapper/striped-pmem
rm -r /pmem

# ------------------------------------------------
#        Check the pmem regions
# ------------------------------------------------
ipmctl show -region


# ------------------------------------------------
#        Check the namespaces
# ------------------------------------------------
ndctl list
ndctl disable-namespace namespace0.0
ndctl disable-namespace namespace1.0
ndctl disable-namespace namespace2.0
ndctl disable-namespace namespace3.0
ndctl destroy-namespace all --force

ndctl list
