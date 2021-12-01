cd input/
wget http://www.diag.uniroma1.it//challenge9/data/USA-road-d/USA-road-d.USA.gr.gz
gunzip USA-road-d.USA.gr.gz
cd ..
cd convertor/
make
./convertor ../input/USA-road-d.USA.gr ../input/USA-road-d.USA.gr.parboil 0 1
