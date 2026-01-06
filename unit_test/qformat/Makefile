# Simple Makefile for the q-format unit tests
# Enable C++ iostreams
qformat_unit_test: qformat_unit_test.cpp
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the unit tests
	g++ -g -Wall -Wextra -Werror -Wpedantic -o ./output/qformat_unit_test qformat_unit_test.cpp -DTINYMIND_ENABLE_OSTREAMS=1 -I ../../include -I../../cpp -I${BOOST_HOME}

run :
#	Run the q-format unit tests
	cd ./output && ./qformat_unit_test && cd ../

# Remove all object files
clean:
	rm -f ./output/*
