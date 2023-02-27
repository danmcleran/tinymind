# Simple Makefile for the q-format unit tests
# Enable C++ iostreams
qformat_unit_test: qformat_unit_test.cpp
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the unit tests
	g++ -g -Wall -o ./output/qformat_unit_test qformat_unit_test.cpp -DTINYMIND_ENABLE_OSTREAMS=1 -I../../cpp -I${BOOST_HOME}

# Remove all object files
clean:
	rm -f *.o
