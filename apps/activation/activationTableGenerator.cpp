/**
* Copyright (c) 2020 Intel Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>

#include "qformat.hpp"
#include "activation.hpp"

namespace fs = std::filesystem; // Must use C++ 17

typedef enum
{
    tanhActivation,
    sigmoidActivation,
    expActivation,
    logActivation,
    sinActivation,
    cosActivation,
    endActivation
} activation_e;

typedef struct
{
    size_t startYear;
    size_t endYear;
} copyrightSpan_t;

static std::string folderPathString;
// The copyright texts sit beside output/, where the build puts the generator,
// so they are found from the generator's own path whatever the working
// directory.
static std::filesystem::path sourceDir;
static char const* const activationTags[] = {"TINYMIND_USE_TANH_", "TINYMIND_USE_SIGMOID_", "TINYMIND_USE_EXP_", "TINYMIND_USE_LOG_", "TINYMIND_USE_SIN_", "TINYMIND_USE_COS_"};
// Table storage widths. QValue caps formats at 64 bits (qformat.hpp), so a
// wider table could never be read; 128-bit tables were generated until #287.
static const uint8_t tables[] = {8, 16, 32, 64};

using namespace std;

static const string lutFileName = "lookupTables.cpp";
static const string selectorPath[] = {"tanh.hpp", "sigmoid.hpp", "exp.hpp", "log.hpp", "sin.hpp", "cos.hpp"};
static const string valuesPathPrefix[] = {"tanhValues", "sigmoidValues", "expValues", "logValues", "sinValues", "cosValues"};
static const string structPrefixes[] = {"Tanh", "Sigmoid", "Exp", "Log", "Sin", "Cos"};
static const string includePrefix[] = {"tanh", "sigmoid", "exp", "log", "sin", "cos"};

static_assert((sizeof(selectorPath) / sizeof(string)) == endActivation, "Invalid selector file path array size.");
static_assert((sizeof(valuesPathPrefix) / sizeof(string)) == endActivation, "Invalid path prefix array size.");
static_assert((sizeof(structPrefixes) / sizeof(string)) == endActivation, "Invalid struct prefix array size.");
static_assert((sizeof(includePrefix) / sizeof(string)) == endActivation, "Invalid include prefix array size.");

static double sigmoid(const double x)
{
    const double exponential = std::exp(x);
    const double result = (exponential / (exponential + static_cast<double>(1.0)));
    
    return result;
}

static bool activationRequiresIntelCopyright(const activation_e activationType)
{
    return (activationType != sinActivation && activationType != cosActivation);
}

static void writeFileCopyrightAndLicense(const string& path, const bool includeIntelCopyright = true)
{
    ifstream myInFile(sourceDir / "my_copyright.txt");
    ofstream outFile(path);
    char buffer[1024];

    assert(myInFile.is_open());

    while(!myInFile.eof())
    {
        myInFile.getline(buffer, 1024);
        outFile.write(buffer, strlen(buffer));
        outFile << std::endl;
    }

    if (includeIntelCopyright)
    {
        ifstream intelInFile(sourceDir / "intel_copyright.txt");
        assert(intelInFile.is_open());

        while(!intelInFile.eof())
        {
            intelInFile.getline(buffer, 1024);
            outFile.write(buffer, strlen(buffer));
            outFile << std::endl;
        }

        intelInFile.close();
    }

    outFile << std::endl;

    outFile.flush();
    outFile.close();
}

static void writeNamespaceBegin(const string& path)
{
    ofstream outFile(path, ofstream::app);

    outFile << "namespace tinymind {" << endl;
}

static void writeNamespaceEnd(const string& path)
{
    ofstream outFile(path, ofstream::app);

    outFile << "}" << endl;
}

static void writeActivationFileHeader(const string& path)
{
    writeFileCopyrightAndLicense(path);

    ofstream outFile(path, ofstream::app);
    outFile << "#pragma once" << endl << endl;
}

static void writeFileHeader(const string& path, const bool includeIntelCopyright = true)
{
    writeFileCopyrightAndLicense(path, includeIntelCopyright);

    ofstream outFile(path, ofstream::app);
    outFile << "#pragma once" << endl << endl;
    outFile << "#include <cstdint>" << endl << endl;
    outFile << "#include \"activation.hpp\"" << endl << endl;
}

// The log table holds log(m) for the mantissa m in [1, 2]; LogLookup in
// logLookup.hpp adds the exponent times ln 2. The sin and cos tables hold one
// quarter wave, [0, pi/2]; TrigLookup in trigLookup.hpp reduces the input by
// the period and mirrors it into that quarter. The activation tables (tanh,
// sigmoid, exp) span [MIN_X_TABLE_VALUE, MAX_X_TABLE_VALUE).
static char const* tableSizeMacro(const activation_e activationType)
{
    if (logActivation == activationType)
    {
        return "NUMBER_OF_LOG_TABLE_VALUES";
    }
    else if ((sinActivation == activationType) || (cosActivation == activationType))
    {
        return "NUMBER_OF_TRIG_TABLE_VALUES";
    }

    return "NUMBER_OF_ACTIVATION_TABLE_VALUES";
}

static int tableSize(const activation_e activationType)
{
    if (logActivation == activationType)
    {
        return NUMBER_OF_LOG_TABLE_VALUES;
    }
    else if ((sinActivation == activationType) || (cosActivation == activationType))
    {
        return NUMBER_OF_TRIG_TABLE_VALUES;
    }

    return NUMBER_OF_ACTIVATION_TABLE_VALUES;
}

static double tableSampleX(const activation_e activationType, const int index)
{
    if (logActivation == activationType)
    {
        return 1.0 + std::ldexp(static_cast<double>(index), -LOG_TABLE_SHIFT);
    }
    else if ((sinActivation == activationType) || (cosActivation == activationType))
    {
        const double halfPi = std::acos(-1.0) / 2.0;
        return halfPi * std::ldexp(static_cast<double>(index), -TRIG_TABLE_SHIFT);
    }

    return static_cast<double>(MIN_X_TABLE_VALUE) + std::ldexp(static_cast<double>(index), -ACTIVATION_DELTA_SHIFT);
}

static void writeTableHeader(const string& path, char const* const preprocessorTag, const int totalBits, const int fixedBits, const int fracBits, const activation_e activationType)
{
    const string spaces("    ");
    ofstream outFile;
    outFile.open(path, ofstream::app);
    outFile << spaces << "#if " << preprocessorTag << fixedBits << "_" << fracBits << endl;
    outFile << spaces << "struct " << structPrefixes[activationType] << "ValuesTableQ" << fixedBits << "_" << fracBits << endl;
    outFile << spaces << "{" << endl;
    outFile << spaces << "    static const uint" << totalBits << "_t values[" << tableSizeMacro(activationType) << "];" << endl;
    outFile << spaces << "};" << endl;
    outFile << spaces << "#endif // "  << preprocessorTag << fixedBits << "_" << fracBits << endl;
    outFile.close();
}

static void writeLutFileHeader(const string& path)
{
    writeFileCopyrightAndLicense(path);

    ofstream outFile(path, ofstream::app);
    outFile << "#include <cstdint>" << endl << endl;
    
    outFile << "#include \"activation.hpp\"" << endl;
    for(int i = 0; i < endActivation;++i)
    {
        for(size_t value : tables)
        {
            outFile << "#include \"" << includePrefix[i] << "Values" << value << "Bit.hpp\"" << endl;
        }
    }
    outFile << endl;
    outFile.close();
}

static void writeTableSelectorTemplate(const string& path, const activation_e activationType)
{
    const string spaces("    ");
    ofstream outFile;
    outFile.open(path, ofstream::app);
    outFile << spaces << "template<unsigned FixedBits, unsigned FracBits, bool IsSigned>\n";
    outFile << spaces << "struct " << structPrefixes[activationType] << "TableValueSize\n";
    outFile << spaces << "{\n";
    outFile << spaces << "};\n\n";
}

static void writeTableSelectorInclude(const string& path, const int numBits, const activation_e activationType)
{
    ofstream outFile;
    outFile.open(path, ofstream::app);
    outFile << "#include \"" << includePrefix[activationType] << "Values" << numBits << "Bit.hpp\"" << endl;
}

static void writeSelectorMinMaxShift(const string& path)
{
    ofstream outFile;
    outFile.open(path, ofstream::app);
    outFile << "#define NUMBER_OF_ACTIVATION_TABLE_VALUES " << NUMBER_OF_ACTIVATION_TABLE_VALUES << endl;
    outFile << "#define MIN_X_TABLE_VALUE (" << MIN_X_TABLE_VALUE << ")" << endl;
    outFile << "#define MAX_X_TABLE_VALUE (" << MAX_X_TABLE_VALUE << ")" << endl;
    outFile << "#define ACTIVATION_DELTA_SHIFT " << ACTIVATION_DELTA_SHIFT << endl;
    outFile << "#define NUMBER_OF_LOG_TABLE_VALUES " << NUMBER_OF_LOG_TABLE_VALUES << endl;
    outFile << "#define LOG_TABLE_SHIFT " << LOG_TABLE_SHIFT << endl;
    outFile << "#define NUMBER_OF_TRIG_TABLE_VALUES " << NUMBER_OF_TRIG_TABLE_VALUES << endl;
    outFile << "#define TRIG_TABLE_SHIFT " << TRIG_TABLE_SHIFT << endl;
    outFile << endl;
}

static void writeTableSelectorCases(const string& path, const int fixedBits, const int fracBits, const activation_e activationType)
{
    const string spaces("    ");
    ofstream outFile;
    outFile.open(path, ofstream::app);
    outFile << spaces << "#if " << activationTags[activationType] << fixedBits << "_" << fracBits << endl;
    outFile << spaces << "template<>" << endl;
    outFile << spaces << "struct " << structPrefixes[activationType] << "TableValueSize<"<< fixedBits << ", " << fracBits << ", true>\n";
    outFile << spaces << "{" << endl;
    outFile << spaces << "    typedef " << structPrefixes[activationType] << "ValuesTableQ" << fixedBits << "_" << fracBits << " " << structPrefixes[activationType] << "TableType;\n";
    outFile << spaces << "};" << endl;
    outFile << spaces << "#endif // "  << activationTags[activationType] << fixedBits << "_" << fracBits << endl << endl;
}

static void writeTableSelector(string path, const activation_e activationType)
{
    const string spaces("    ");
    ofstream outFile;
    outFile.open(path, ofstream::app);
    outFile << spaces << "template<unsigned FixedBits,unsigned FracBits, bool IsSigned>\n";
    outFile << spaces << "struct " << structPrefixes[activationType] << "ValuesTableSelector\n";
    outFile << spaces << "{" << endl;
    outFile << spaces << "    typedef typename " << structPrefixes[activationType] << "TableValueSize<FixedBits, FracBits, IsSigned>::" << structPrefixes[activationType] <<
                "TableType " << structPrefixes[activationType] << "TableType;\n";
    outFile << spaces << "};" << endl;
}

// The raw value of `num` (the function value times 2^fracBits) in a signed
// format `totalBits` wide: truncated toward zero, as the tables always were,
// and saturated to the format's range. Casting an out-of-range double to an
// integer is undefined; on x86 it wrapped, so exp(x) for x >= 4.875 came out
// negative in Q8.8 and cos(0) came out as -1 in Q1.15.
static int64_t toSaturatedFixed64(const double num, const unsigned totalBits)
{
    const double limit = std::ldexp(1.0, static_cast<int>(totalBits) - 1);
    const int64_t maxValue = static_cast<int64_t>((static_cast<uint64_t>(1) << (totalBits - 1)) - 1);

    if (num >= limit)
    {
        return maxValue;
    }
    else if (num < -limit)
    {
        return -maxValue - 1;
    }

    return static_cast<int64_t>(num);
}

template <typename T>
static void writeLutValues(string path, const size_t totalBits, uint64_t fixedBits, uint64_t fracBits, const activation_e activationType)
{
    const string spaces("    ");
    ofstream outFile;
    outFile.open(path, ofstream::app);
    double activate;
    double num;

    outFile << std::dec;
    outFile << spaces << "#if " << activationTags[activationType] << fixedBits << "_" << fracBits << endl;
    outFile << spaces << "const uint" << totalBits << "_t " << structPrefixes[activationType] << "ValuesTableQ" << fixedBits << "_" << fracBits << "::values[" << tableSizeMacro(activationType) << "] = {" << endl;

    for (int i = 0; i < tableSize(activationType); ++i)
    {
        const double value = tableSampleX(activationType, i);

        if (tanhActivation == activationType)
        {
            activate = tanh(value);
        }
        else if (sigmoidActivation == activationType)
        {
            activate = sigmoid(value);
        }
        else if (expActivation == activationType)
        {
            activate = exp(value);
        }
        else if (logActivation == activationType)
        {
            activate = log(value);
        }
        else if (sinActivation == activationType)
        {
            activate = std::sin(value);
        }
        else if (cosActivation == activationType)
        {
            activate = std::cos(value);
        }
        else
        {
            assert(false);
        }

        // Handle the bit shift calculation carefully for large fracBits
        double leftShift;
        if (fracBits <= 63)
        {
            leftShift = static_cast<double>(1ULL << fracBits);
        } else {
            // For fracBits > 63, calculate 2^fracBits using pow
            leftShift = pow(2.0, static_cast<double>(fracBits));
        }
        
        num = activate * leftShift;

        switch (std::numeric_limits<T>::digits)
        {
            case 8:
                outFile << spaces << "            0x" << std::hex << std::uppercase << (int)(uint8_t)toSaturatedFixed64(num, 8) << "," << endl; // cast to uint8_t first for the correct value. cast next to int to write to file correctly as a number not a char.
                break;

            case 16:
                outFile << spaces << "            0x" << std::hex << std::uppercase << (uint16_t)toSaturatedFixed64(num, 16) << "," << endl;
                break;

            case 32:
                outFile << spaces << "            0x" << std::hex << std::uppercase << (uint32_t)toSaturatedFixed64(num, 32) << "," << endl;
                break;

            case 64:
                outFile << spaces << "            0x" << std::hex << std::uppercase << (uint64_t)toSaturatedFixed64(num, 64) << "," << endl;
                break;

            default:
                assert(0);
        }
    }

    outFile << std::dec;
    outFile << spaces << "};" << endl;
    outFile << spaces << "#endif // "  << activationTags[activationType] << fixedBits << "_" << fracBits << endl;
}

static void generateLut(const string& path, const activation_e activationType)
{
    for(uint8_t index = 0;index < ((sizeof(tables) / sizeof(tables[0])));++index)
    {
        const uint8_t totalBits = tables[index];

        for (int i = 1; i < totalBits; i++)
        {
            switch (totalBits)
            {
            case 8:
                writeLutValues<uint8_t>(path, totalBits, i, totalBits - i, activationType);
                break;

            case 16:
                writeLutValues<uint16_t>(path, totalBits, i, totalBits - i, activationType);
                break;

            case 32:
                writeLutValues<uint32_t>(path, totalBits, i, totalBits - i, activationType);
                break;

            case 64:
                writeLutValues<uint64_t>(path, totalBits, i, totalBits - i, activationType);
                break;

            default:
                assert(0);
            }
        }
    }
}

static void generateHeader(const activation_e activationType)
{
    fs::path selectorFilePath(folderPathString);
    const bool includeIntelCopyright = activationRequiresIntelCopyright(activationType);

    selectorFilePath /= selectorPath[activationType];

    for(uint8_t index = 0;index < ((sizeof(tables) / sizeof(tables[0])));++index)
    {
        fs::path filePath(folderPathString);
        const uint8_t totalBits = tables[index];
        string fileName = valuesPathPrefix[activationType] + to_string(totalBits);

        fileName.append("Bit.hpp");

        filePath /= fileName;

        writeFileHeader(filePath.string(), includeIntelCopyright);

        writeNamespaceBegin(filePath.string());

        for (int i = 1; i < totalBits; i++)
        {
            writeTableHeader(filePath.string(), activationTags[activationType], totalBits, i, totalBits - i, activationType);
        }

        writeNamespaceEnd(filePath.string());
    }


    writeFileHeader(selectorFilePath.string(), includeIntelCopyright);

    for(uint8_t index = 0;index < ((sizeof(tables) / sizeof(tables[0])));++index)
    {
        const uint8_t totalBits = tables[index];
        writeTableSelectorInclude(selectorFilePath.string(), totalBits, activationType);
    }

    {
        ofstream outFile(selectorFilePath.string(), ofstream::app);
        outFile << endl;
    }

    writeNamespaceBegin(selectorFilePath.string());

    writeTableSelectorTemplate(selectorFilePath.string(), activationType);

    for(uint8_t index = 0;index < ((sizeof(tables) / sizeof(tables[0])));++index)
    {
        const uint8_t totalBits = tables[index];
        
        for (int i = 1; i < totalBits; i++)
        {
            switch (totalBits)
            {
            case 8:
                writeTableSelectorCases(selectorFilePath.string(), i, totalBits - i, activationType);
                break;

            case 16:
                writeTableSelectorCases(selectorFilePath.string(), i, totalBits - i, activationType);
                break;

            case 32:
                writeTableSelectorCases(selectorFilePath.string(), i, totalBits - i, activationType);
                break;

            case 64:
                writeTableSelectorCases(selectorFilePath.string(), i, totalBits - i, activationType);
                break;

            default:
                assert(0);
            }
        }
    }

    writeTableSelector(selectorFilePath.string(), activationType);

    writeNamespaceEnd(selectorFilePath.string());
}

int main(const int argc, char *argv[])
{
    if(argc != 2)
    {
        cerr << "Must provide folder path for output files." << endl;
        return -1;
    }

    folderPathString = argv[1];
    sourceDir = fs::weakly_canonical(fs::absolute(argv[0])).parent_path().parent_path();

    fs::path folderPath(folderPathString);

    fs::path activationHeaderPath(folderPath);
    activationHeaderPath /= "activation.hpp";

    writeActivationFileHeader(activationHeaderPath.string());
    writeSelectorMinMaxShift(activationHeaderPath.string());

    generateHeader(sigmoidActivation);
    generateHeader(tanhActivation);
    generateHeader(expActivation);
    generateHeader(logActivation);
    generateHeader(sinActivation);
    generateHeader(cosActivation);

    fs::path lutPath(folderPath);
    lutPath /= lutFileName;

    writeLutFileHeader(lutPath.string());

    writeNamespaceBegin(lutPath.string());

    for(int activation = tanhActivation;activation < endActivation;++activation)
    {
        generateLut(lutPath.string(), (activation_e)activation);
    }

    writeNamespaceEnd(lutPath.string());
}

