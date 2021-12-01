# bash script to configure the Optane persistent memory

# ------------------------------------------------
#        Check the pmem regions after reboot
# ------------------------------------------------
ipmctl show -region


# ------------------------------------------------
#        Check the namespaces after reboot
# ------------------------------------------------
ndctl list

ndctl create-namespace --mode=fsdax
ndctl create-namespace --mode=fsdax
ndctl create-namespace --mode=fsdax
ndctl create-namespace --mode=fsdax

ndctl list

# lsblk
# would list all the 3 newly created namespaces with fsdax mode as new blk devices: pmem0, pmem1, pmem2, pmem3

# ------------------------------------------------
#        Check the new block devices created
# ------------------------------------------------
# All 3 newly created namespaces are created as new blk devices
ls -l /dev/pmem*
# Output will contain: /dev/pmem0, /dev/pmem1, /dev/pmem2, /dev/pmem3

echo -e "0 $(( `blockdev --getsz /dev/pmem0` + `blockdev --getsz /dev/pmem1` + `blockdev --getsz /dev/pmem2` + `blockdev --getsz /dev/pmem3` )) striped 4 16 /dev/pmem0 0 /dev/pmem1 0 /dev/pmem2 0 /dev/pmem3 0" | sudo dmsetup create striped-pmem

lsblk /dev/pmem*

echo "g\nn\n\n\n\nw\n" | fdisk /dev/mapper/striped-pmem

# ------------------------------------------------
#        Setting up filesystem on the pmem block devices
# ------------------------------------------------
mkfs.ext4 -b 4096 -E stride=512 -F /dev/mapper/striped-pmem

# ------------------------------------------------
#        Attaching the devices to the root file system - create mount points
# ------------------------------------------------
mkdir -p /pmem

# ------------------------------------------------
#        Mount the devices to the created mount points with -o dax option
# ------------------------------------------------
mount -o dax /dev/mapper/striped-pmem /pmem

chmod a+w /pmem

# lsblk command now will show all 3 devices with the mount points
#----- Output of lsblk ---------
# pmem2       259:5    0 248.1G  0 disk /pmem/


# ------------------------------------------------
#        Check the namespaces
# ------------------------------------------------
sudo ndctl list

