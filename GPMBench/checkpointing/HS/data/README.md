Hotspot input files come in pairs. One file represents power data and one
represents temperature data. They each represent a square matrix of values. The
number in the filename is the number of values on a side of the matrix. For
example, temp_64 represents a 64 x 64 matrix.

The files temp_1024  and power_1024 are prexisting input files of real data. To obtain inputs of different sizes, expand these using hotspotex, and verify their correctness using hotspotver. To
run these programs, edit hotspotex.cpp and hotspotver.cpp to uncomment the
appropriate header file, make, and run them. Headers are provided for input
sizes of powers of 2 up to 16384.
