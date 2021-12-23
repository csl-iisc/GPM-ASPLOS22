GSL_VER = 1.14

all:
	mkdir -p release/db release/reader release/sql release/table release/tablet release/vm
	make -C src


TYPES ?= _gpm _mm_gpu _fs_gpu _gpm_rest
BUILD ?= ./binaries/

run_all:
	mkdir -p ../results;
	for i in ${TYPES}; do \
		sudo PMEM_THREADS=8 ${BUILD}/gpDB$$i comparedb 50000000 > ../results/gpDB$$i.txt; \
	done

run_rest:
	mkdir -p ../results;
	for i in ${TYPES}; do \
		sudo PMEM_THREADS=8 ${BUILD}/gpDB$$i comparedb 50000000 > ../results/gpDB$$i.txt; \
	done
	
run_fig_10:
	sudo ${BUILD}/gpDB_gpm_ndp comparedb 50000000 > ../results/gpDB_gpm_ndp.txt;
	
run_fig_11a:
	sudo ${BUILD}/gpDB_gpm comparedb 50000000 > ../results/gpDB_gpm.txt;
	sudo ${BUILD}/gpDB_gpm_conv comparedb 50000000 > ../results/gpDB_gpm_conv.txt;

clean:
	rm -rf gsl gsl-$(GSL_VER) db/comparedb virginian.tar.gz debug release
	mkdir -p doc
	make -C example clean
	make -C db clean
	make -C src clean

gsl:
	rm -rf gsl-$(GSL_VER) gsl lib/libgsl.so lib/libgslcblas.so lib/gsl
	wget ftp://ftp.gnu.org/gnu/gsl/gsl-$(GSL_VER).tar.gz
	tar -xzzvf gsl-$(GSL_VER).tar.gz
	rm gsl-$(GSL_VER).tar.gz
	mkdir gsl
	cd gsl && ../gsl-$(GSL_VER)/configure
	cd gsl && make
	cp gsl/cblas/.libs/libgslcblas.so lib
	cp gsl/.libs/libgsl.so lib
	cp -rL gsl/gsl/ lib/
	rm -rf gsl-$(GSL_VER) gsl lib/gsl/Makefile

gtest:
	rm -rf lib/libgtest.a lib/gtest
	wget https://github.com/google/googletest/archive/release-1.6.0.zip
	unzip release-1.6.0.zip
	mv googletest-release-1.6.0 gtest
	rm release-1.6.0.zip
	g++ -g3 -Igtest/include -Igtest -c gtest/src/gtest-all.cc -o lib/gtest-all.o
	ar -rv lib/libgtest.a lib/gtest-all.o
	cp -rL gtest/include/gtest lib/
	rm lib/gtest-all.o
	rm -rf gtest

package:
	make -C src clean
	tar czvf virginian.tar.gz ../virginian/db/Makefile ../virginian/db/generate.c ../virginian/lib/ ../virginian/debug/ ../virginian/release.gcc/ ../virginian/release.icc/ ../virginian/doc/ ../virginian/src/ ../virginian/example/ ../virginian/Makefile ../virginian/README

.PHONY: package
.PHONY: gsl
.PHONY: gtest

