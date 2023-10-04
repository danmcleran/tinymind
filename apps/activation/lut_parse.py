import os
import sys
import argparse
import ctypes
import matplotlib.pyplot as pyplot

if __name__ == "__main__":
    lutFilePath = r'..' + os.sep + '..' + os.sep + 'cpp' + os.sep + 'lookupTables.cpp'
    assert(os.path.exists(lutFilePath))
    supportedFns = ['tanh', 'log', 'sigmoid']
    supportedTotalBits = [8,16,32,64]
    values = []
    parser = argparse.ArgumentParser(prog='LUT Parser', description='Parse and plot activation function LUT(s)')
    parser.add_argument('function', type=str, help='Specify an activation function (e.g., tanh, etc.).')
    parser.add_argument('qformat', type=str, help='Specify Q-format to parse and plot (e.g., 8.8, 24.8, 2.7, etc.).')
    args = parser.parse_args()

    activationFn = args.function
    if not activationFn in supportedFns:
        print('Unsupported activation function: %s' % activationFn)
        print('Supported activation functions are:')
        [print(fn) for fn in supportedFns]
        sys.exit(-1)

    qformatSplit = args.qformat.split('.')
    fixedBits = int(qformatSplit[0])
    fractionalBits = int(qformatSplit[1])
    totalNumBits = (fixedBits + fractionalBits)
    if not totalNumBits in supportedTotalBits:
        print('Unsupported number of bits: %d' % totalNumBits)
        print('Supported bits consists of:')
        [print(value) for value in supportedTotalBits]
        sys.exit(-1)

    print('Parsing values from %s' % lutFilePath)
    print("Parsing the %s activation function for Q%d.%d" % (activationFn, fixedBits, fractionalBits))
    buildSwitch = "TINYMIND_USE_%s_%d_%d" % (activationFn.upper(), int(qformatSplit[0]), int(qformatSplit[1]))
    print("Build switch: %s" % buildSwitch)
    searchString = "#if %s" % buildSwitch
    found= False
    parse = False
    with open(lutFilePath, 'r') as f:
        for line in f.readlines():
            if not found:
                if searchString in line:
                    found = True
            else:
                if not parse:
                    parse = True
                else:
                    if '}' in line:
                        break
                    if totalNumBits == 8:
                        value = ctypes.c_int8(int(line.split(',')[0], 16)).value
                    elif totalNumBits == 16:
                        value = ctypes.c_int16(int(line.split(',')[0], 16)).value
                    elif totalNumBits == 32:
                        value = ctypes.c_int32(int(line.split(',')[0], 16)).value
                    elif totalNumBits == 64:
                        value = ctypes.c_int64(int(line.split(',')[0], 16)).value
                    values.append(value)

    if not len(values):
        print("Failed to parse out LUT values for activation function: %s, fixed bits: %d, and fractional bits: %d" % (activationFn, fixedBits, fractionalBits))
        sys.exit(-1)

    pyplot.figure()
    pyplot.plot(values, 'b-x')
    pyplot.title('%s - Q%d.%d' % (activationFn, fixedBits, fractionalBits))
    pyplot.xlabel('LUT index')
    pyplot.ylabel('LUT value')
    pyplot.grid()
    pyplot.show()