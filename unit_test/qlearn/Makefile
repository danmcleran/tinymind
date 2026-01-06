# Simple Makefile for the q-learn unit tests
# Enable C++ iostreams
# Tell the code to compile the Q16.16 tanh activation function LUT
qlearn_unit_test: qlearn_unit_test.cpp ../../cpp/lookupTables.cpp
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the unit tests
	g++ -g -Wall -Wextra -Werror -o ./output/qlearn_unit_test qlearn_unit_test.cpp ../../cpp/lookupTables.cpp -DTINYMIND_ENABLE_OSTREAMS=1 -I../../cpp -I../../cpp/include -I../../include -DTINYMIND_USE_TANH_16_16=1 -I${BOOST_HOME}

run :
#	Run the q-learn unit tests
	cd ./output && ./qlearn_unit_test && cd ../

# Remove all object files
clean:
	rm -f ./output/*
