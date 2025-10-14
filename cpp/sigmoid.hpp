/**
* Copyright (c) 2023 Dan McLeran
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

#pragma once

#include "activation.hpp"

#include "sigmoidValues8Bit.hpp"
#include "sigmoidValues16Bit.hpp"
#include "sigmoidValues32Bit.hpp"
#include "sigmoidValues64Bit.hpp"
#include "sigmoidValues128Bit.hpp"

namespace tinymind {
    template<unsigned FixedBits, unsigned FracBits, bool IsSigned>
    struct SigmoidTableValueSize
    {
    };

    #if TINYMIND_USE_SIGMOID_1_7
    template<>
    struct SigmoidTableValueSize<1, 7, true>
    {
        typedef SigmoidValuesTableQ1_7 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_1_7

    #if TINYMIND_USE_SIGMOID_2_6
    template<>
    struct SigmoidTableValueSize<2, 6, true>
    {
        typedef SigmoidValuesTableQ2_6 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_2_6

    #if TINYMIND_USE_SIGMOID_3_5
    template<>
    struct SigmoidTableValueSize<3, 5, true>
    {
        typedef SigmoidValuesTableQ3_5 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_3_5

    #if TINYMIND_USE_SIGMOID_4_4
    template<>
    struct SigmoidTableValueSize<4, 4, true>
    {
        typedef SigmoidValuesTableQ4_4 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_4_4

    #if TINYMIND_USE_SIGMOID_5_3
    template<>
    struct SigmoidTableValueSize<5, 3, true>
    {
        typedef SigmoidValuesTableQ5_3 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_5_3

    #if TINYMIND_USE_SIGMOID_6_2
    template<>
    struct SigmoidTableValueSize<6, 2, true>
    {
        typedef SigmoidValuesTableQ6_2 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_6_2

    #if TINYMIND_USE_SIGMOID_7_1
    template<>
    struct SigmoidTableValueSize<7, 1, true>
    {
        typedef SigmoidValuesTableQ7_1 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_7_1

    #if TINYMIND_USE_SIGMOID_1_15
    template<>
    struct SigmoidTableValueSize<1, 15, true>
    {
        typedef SigmoidValuesTableQ1_15 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_1_15

    #if TINYMIND_USE_SIGMOID_2_14
    template<>
    struct SigmoidTableValueSize<2, 14, true>
    {
        typedef SigmoidValuesTableQ2_14 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_2_14

    #if TINYMIND_USE_SIGMOID_3_13
    template<>
    struct SigmoidTableValueSize<3, 13, true>
    {
        typedef SigmoidValuesTableQ3_13 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_3_13

    #if TINYMIND_USE_SIGMOID_4_12
    template<>
    struct SigmoidTableValueSize<4, 12, true>
    {
        typedef SigmoidValuesTableQ4_12 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_4_12

    #if TINYMIND_USE_SIGMOID_5_11
    template<>
    struct SigmoidTableValueSize<5, 11, true>
    {
        typedef SigmoidValuesTableQ5_11 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_5_11

    #if TINYMIND_USE_SIGMOID_6_10
    template<>
    struct SigmoidTableValueSize<6, 10, true>
    {
        typedef SigmoidValuesTableQ6_10 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_6_10

    #if TINYMIND_USE_SIGMOID_7_9
    template<>
    struct SigmoidTableValueSize<7, 9, true>
    {
        typedef SigmoidValuesTableQ7_9 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_7_9

    #if TINYMIND_USE_SIGMOID_8_8
    template<>
    struct SigmoidTableValueSize<8, 8, true>
    {
        typedef SigmoidValuesTableQ8_8 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_8_8

    #if TINYMIND_USE_SIGMOID_9_7
    template<>
    struct SigmoidTableValueSize<9, 7, true>
    {
        typedef SigmoidValuesTableQ9_7 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_9_7

    #if TINYMIND_USE_SIGMOID_10_6
    template<>
    struct SigmoidTableValueSize<10, 6, true>
    {
        typedef SigmoidValuesTableQ10_6 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_10_6

    #if TINYMIND_USE_SIGMOID_11_5
    template<>
    struct SigmoidTableValueSize<11, 5, true>
    {
        typedef SigmoidValuesTableQ11_5 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_11_5

    #if TINYMIND_USE_SIGMOID_12_4
    template<>
    struct SigmoidTableValueSize<12, 4, true>
    {
        typedef SigmoidValuesTableQ12_4 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_12_4

    #if TINYMIND_USE_SIGMOID_13_3
    template<>
    struct SigmoidTableValueSize<13, 3, true>
    {
        typedef SigmoidValuesTableQ13_3 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_13_3

    #if TINYMIND_USE_SIGMOID_14_2
    template<>
    struct SigmoidTableValueSize<14, 2, true>
    {
        typedef SigmoidValuesTableQ14_2 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_14_2

    #if TINYMIND_USE_SIGMOID_15_1
    template<>
    struct SigmoidTableValueSize<15, 1, true>
    {
        typedef SigmoidValuesTableQ15_1 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_15_1

    #if TINYMIND_USE_SIGMOID_1_31
    template<>
    struct SigmoidTableValueSize<1, 31, true>
    {
        typedef SigmoidValuesTableQ1_31 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_1_31

    #if TINYMIND_USE_SIGMOID_2_30
    template<>
    struct SigmoidTableValueSize<2, 30, true>
    {
        typedef SigmoidValuesTableQ2_30 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_2_30

    #if TINYMIND_USE_SIGMOID_3_29
    template<>
    struct SigmoidTableValueSize<3, 29, true>
    {
        typedef SigmoidValuesTableQ3_29 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_3_29

    #if TINYMIND_USE_SIGMOID_4_28
    template<>
    struct SigmoidTableValueSize<4, 28, true>
    {
        typedef SigmoidValuesTableQ4_28 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_4_28

    #if TINYMIND_USE_SIGMOID_5_27
    template<>
    struct SigmoidTableValueSize<5, 27, true>
    {
        typedef SigmoidValuesTableQ5_27 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_5_27

    #if TINYMIND_USE_SIGMOID_6_26
    template<>
    struct SigmoidTableValueSize<6, 26, true>
    {
        typedef SigmoidValuesTableQ6_26 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_6_26

    #if TINYMIND_USE_SIGMOID_7_25
    template<>
    struct SigmoidTableValueSize<7, 25, true>
    {
        typedef SigmoidValuesTableQ7_25 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_7_25

    #if TINYMIND_USE_SIGMOID_8_24
    template<>
    struct SigmoidTableValueSize<8, 24, true>
    {
        typedef SigmoidValuesTableQ8_24 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_8_24

    #if TINYMIND_USE_SIGMOID_9_23
    template<>
    struct SigmoidTableValueSize<9, 23, true>
    {
        typedef SigmoidValuesTableQ9_23 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_9_23

    #if TINYMIND_USE_SIGMOID_10_22
    template<>
    struct SigmoidTableValueSize<10, 22, true>
    {
        typedef SigmoidValuesTableQ10_22 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_10_22

    #if TINYMIND_USE_SIGMOID_11_21
    template<>
    struct SigmoidTableValueSize<11, 21, true>
    {
        typedef SigmoidValuesTableQ11_21 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_11_21

    #if TINYMIND_USE_SIGMOID_12_20
    template<>
    struct SigmoidTableValueSize<12, 20, true>
    {
        typedef SigmoidValuesTableQ12_20 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_12_20

    #if TINYMIND_USE_SIGMOID_13_19
    template<>
    struct SigmoidTableValueSize<13, 19, true>
    {
        typedef SigmoidValuesTableQ13_19 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_13_19

    #if TINYMIND_USE_SIGMOID_14_18
    template<>
    struct SigmoidTableValueSize<14, 18, true>
    {
        typedef SigmoidValuesTableQ14_18 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_14_18

    #if TINYMIND_USE_SIGMOID_15_17
    template<>
    struct SigmoidTableValueSize<15, 17, true>
    {
        typedef SigmoidValuesTableQ15_17 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_15_17

    #if TINYMIND_USE_SIGMOID_16_16
    template<>
    struct SigmoidTableValueSize<16, 16, true>
    {
        typedef SigmoidValuesTableQ16_16 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_16_16

    #if TINYMIND_USE_SIGMOID_17_15
    template<>
    struct SigmoidTableValueSize<17, 15, true>
    {
        typedef SigmoidValuesTableQ17_15 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_17_15

    #if TINYMIND_USE_SIGMOID_18_14
    template<>
    struct SigmoidTableValueSize<18, 14, true>
    {
        typedef SigmoidValuesTableQ18_14 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_18_14

    #if TINYMIND_USE_SIGMOID_19_13
    template<>
    struct SigmoidTableValueSize<19, 13, true>
    {
        typedef SigmoidValuesTableQ19_13 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_19_13

    #if TINYMIND_USE_SIGMOID_20_12
    template<>
    struct SigmoidTableValueSize<20, 12, true>
    {
        typedef SigmoidValuesTableQ20_12 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_20_12

    #if TINYMIND_USE_SIGMOID_21_11
    template<>
    struct SigmoidTableValueSize<21, 11, true>
    {
        typedef SigmoidValuesTableQ21_11 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_21_11

    #if TINYMIND_USE_SIGMOID_22_10
    template<>
    struct SigmoidTableValueSize<22, 10, true>
    {
        typedef SigmoidValuesTableQ22_10 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_22_10

    #if TINYMIND_USE_SIGMOID_23_9
    template<>
    struct SigmoidTableValueSize<23, 9, true>
    {
        typedef SigmoidValuesTableQ23_9 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_23_9

    #if TINYMIND_USE_SIGMOID_24_8
    template<>
    struct SigmoidTableValueSize<24, 8, true>
    {
        typedef SigmoidValuesTableQ24_8 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_24_8

    #if TINYMIND_USE_SIGMOID_25_7
    template<>
    struct SigmoidTableValueSize<25, 7, true>
    {
        typedef SigmoidValuesTableQ25_7 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_25_7

    #if TINYMIND_USE_SIGMOID_26_6
    template<>
    struct SigmoidTableValueSize<26, 6, true>
    {
        typedef SigmoidValuesTableQ26_6 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_26_6

    #if TINYMIND_USE_SIGMOID_27_5
    template<>
    struct SigmoidTableValueSize<27, 5, true>
    {
        typedef SigmoidValuesTableQ27_5 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_27_5

    #if TINYMIND_USE_SIGMOID_28_4
    template<>
    struct SigmoidTableValueSize<28, 4, true>
    {
        typedef SigmoidValuesTableQ28_4 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_28_4

    #if TINYMIND_USE_SIGMOID_29_3
    template<>
    struct SigmoidTableValueSize<29, 3, true>
    {
        typedef SigmoidValuesTableQ29_3 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_29_3

    #if TINYMIND_USE_SIGMOID_30_2
    template<>
    struct SigmoidTableValueSize<30, 2, true>
    {
        typedef SigmoidValuesTableQ30_2 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_30_2

    #if TINYMIND_USE_SIGMOID_31_1
    template<>
    struct SigmoidTableValueSize<31, 1, true>
    {
        typedef SigmoidValuesTableQ31_1 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_31_1

    #if TINYMIND_USE_SIGMOID_1_63
    template<>
    struct SigmoidTableValueSize<1, 63, true>
    {
        typedef SigmoidValuesTableQ1_63 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_1_63

    #if TINYMIND_USE_SIGMOID_2_62
    template<>
    struct SigmoidTableValueSize<2, 62, true>
    {
        typedef SigmoidValuesTableQ2_62 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_2_62

    #if TINYMIND_USE_SIGMOID_3_61
    template<>
    struct SigmoidTableValueSize<3, 61, true>
    {
        typedef SigmoidValuesTableQ3_61 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_3_61

    #if TINYMIND_USE_SIGMOID_4_60
    template<>
    struct SigmoidTableValueSize<4, 60, true>
    {
        typedef SigmoidValuesTableQ4_60 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_4_60

    #if TINYMIND_USE_SIGMOID_5_59
    template<>
    struct SigmoidTableValueSize<5, 59, true>
    {
        typedef SigmoidValuesTableQ5_59 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_5_59

    #if TINYMIND_USE_SIGMOID_6_58
    template<>
    struct SigmoidTableValueSize<6, 58, true>
    {
        typedef SigmoidValuesTableQ6_58 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_6_58

    #if TINYMIND_USE_SIGMOID_7_57
    template<>
    struct SigmoidTableValueSize<7, 57, true>
    {
        typedef SigmoidValuesTableQ7_57 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_7_57

    #if TINYMIND_USE_SIGMOID_8_56
    template<>
    struct SigmoidTableValueSize<8, 56, true>
    {
        typedef SigmoidValuesTableQ8_56 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_8_56

    #if TINYMIND_USE_SIGMOID_9_55
    template<>
    struct SigmoidTableValueSize<9, 55, true>
    {
        typedef SigmoidValuesTableQ9_55 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_9_55

    #if TINYMIND_USE_SIGMOID_10_54
    template<>
    struct SigmoidTableValueSize<10, 54, true>
    {
        typedef SigmoidValuesTableQ10_54 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_10_54

    #if TINYMIND_USE_SIGMOID_11_53
    template<>
    struct SigmoidTableValueSize<11, 53, true>
    {
        typedef SigmoidValuesTableQ11_53 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_11_53

    #if TINYMIND_USE_SIGMOID_12_52
    template<>
    struct SigmoidTableValueSize<12, 52, true>
    {
        typedef SigmoidValuesTableQ12_52 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_12_52

    #if TINYMIND_USE_SIGMOID_13_51
    template<>
    struct SigmoidTableValueSize<13, 51, true>
    {
        typedef SigmoidValuesTableQ13_51 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_13_51

    #if TINYMIND_USE_SIGMOID_14_50
    template<>
    struct SigmoidTableValueSize<14, 50, true>
    {
        typedef SigmoidValuesTableQ14_50 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_14_50

    #if TINYMIND_USE_SIGMOID_15_49
    template<>
    struct SigmoidTableValueSize<15, 49, true>
    {
        typedef SigmoidValuesTableQ15_49 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_15_49

    #if TINYMIND_USE_SIGMOID_16_48
    template<>
    struct SigmoidTableValueSize<16, 48, true>
    {
        typedef SigmoidValuesTableQ16_48 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_16_48

    #if TINYMIND_USE_SIGMOID_17_47
    template<>
    struct SigmoidTableValueSize<17, 47, true>
    {
        typedef SigmoidValuesTableQ17_47 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_17_47

    #if TINYMIND_USE_SIGMOID_18_46
    template<>
    struct SigmoidTableValueSize<18, 46, true>
    {
        typedef SigmoidValuesTableQ18_46 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_18_46

    #if TINYMIND_USE_SIGMOID_19_45
    template<>
    struct SigmoidTableValueSize<19, 45, true>
    {
        typedef SigmoidValuesTableQ19_45 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_19_45

    #if TINYMIND_USE_SIGMOID_20_44
    template<>
    struct SigmoidTableValueSize<20, 44, true>
    {
        typedef SigmoidValuesTableQ20_44 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_20_44

    #if TINYMIND_USE_SIGMOID_21_43
    template<>
    struct SigmoidTableValueSize<21, 43, true>
    {
        typedef SigmoidValuesTableQ21_43 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_21_43

    #if TINYMIND_USE_SIGMOID_22_42
    template<>
    struct SigmoidTableValueSize<22, 42, true>
    {
        typedef SigmoidValuesTableQ22_42 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_22_42

    #if TINYMIND_USE_SIGMOID_23_41
    template<>
    struct SigmoidTableValueSize<23, 41, true>
    {
        typedef SigmoidValuesTableQ23_41 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_23_41

    #if TINYMIND_USE_SIGMOID_24_40
    template<>
    struct SigmoidTableValueSize<24, 40, true>
    {
        typedef SigmoidValuesTableQ24_40 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_24_40

    #if TINYMIND_USE_SIGMOID_25_39
    template<>
    struct SigmoidTableValueSize<25, 39, true>
    {
        typedef SigmoidValuesTableQ25_39 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_25_39

    #if TINYMIND_USE_SIGMOID_26_38
    template<>
    struct SigmoidTableValueSize<26, 38, true>
    {
        typedef SigmoidValuesTableQ26_38 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_26_38

    #if TINYMIND_USE_SIGMOID_27_37
    template<>
    struct SigmoidTableValueSize<27, 37, true>
    {
        typedef SigmoidValuesTableQ27_37 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_27_37

    #if TINYMIND_USE_SIGMOID_28_36
    template<>
    struct SigmoidTableValueSize<28, 36, true>
    {
        typedef SigmoidValuesTableQ28_36 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_28_36

    #if TINYMIND_USE_SIGMOID_29_35
    template<>
    struct SigmoidTableValueSize<29, 35, true>
    {
        typedef SigmoidValuesTableQ29_35 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_29_35

    #if TINYMIND_USE_SIGMOID_30_34
    template<>
    struct SigmoidTableValueSize<30, 34, true>
    {
        typedef SigmoidValuesTableQ30_34 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_30_34

    #if TINYMIND_USE_SIGMOID_31_33
    template<>
    struct SigmoidTableValueSize<31, 33, true>
    {
        typedef SigmoidValuesTableQ31_33 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_31_33

    #if TINYMIND_USE_SIGMOID_32_32
    template<>
    struct SigmoidTableValueSize<32, 32, true>
    {
        typedef SigmoidValuesTableQ32_32 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_32_32

    #if TINYMIND_USE_SIGMOID_33_31
    template<>
    struct SigmoidTableValueSize<33, 31, true>
    {
        typedef SigmoidValuesTableQ33_31 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_33_31

    #if TINYMIND_USE_SIGMOID_34_30
    template<>
    struct SigmoidTableValueSize<34, 30, true>
    {
        typedef SigmoidValuesTableQ34_30 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_34_30

    #if TINYMIND_USE_SIGMOID_35_29
    template<>
    struct SigmoidTableValueSize<35, 29, true>
    {
        typedef SigmoidValuesTableQ35_29 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_35_29

    #if TINYMIND_USE_SIGMOID_36_28
    template<>
    struct SigmoidTableValueSize<36, 28, true>
    {
        typedef SigmoidValuesTableQ36_28 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_36_28

    #if TINYMIND_USE_SIGMOID_37_27
    template<>
    struct SigmoidTableValueSize<37, 27, true>
    {
        typedef SigmoidValuesTableQ37_27 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_37_27

    #if TINYMIND_USE_SIGMOID_38_26
    template<>
    struct SigmoidTableValueSize<38, 26, true>
    {
        typedef SigmoidValuesTableQ38_26 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_38_26

    #if TINYMIND_USE_SIGMOID_39_25
    template<>
    struct SigmoidTableValueSize<39, 25, true>
    {
        typedef SigmoidValuesTableQ39_25 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_39_25

    #if TINYMIND_USE_SIGMOID_40_24
    template<>
    struct SigmoidTableValueSize<40, 24, true>
    {
        typedef SigmoidValuesTableQ40_24 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_40_24

    #if TINYMIND_USE_SIGMOID_41_23
    template<>
    struct SigmoidTableValueSize<41, 23, true>
    {
        typedef SigmoidValuesTableQ41_23 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_41_23

    #if TINYMIND_USE_SIGMOID_42_22
    template<>
    struct SigmoidTableValueSize<42, 22, true>
    {
        typedef SigmoidValuesTableQ42_22 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_42_22

    #if TINYMIND_USE_SIGMOID_43_21
    template<>
    struct SigmoidTableValueSize<43, 21, true>
    {
        typedef SigmoidValuesTableQ43_21 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_43_21

    #if TINYMIND_USE_SIGMOID_44_20
    template<>
    struct SigmoidTableValueSize<44, 20, true>
    {
        typedef SigmoidValuesTableQ44_20 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_44_20

    #if TINYMIND_USE_SIGMOID_45_19
    template<>
    struct SigmoidTableValueSize<45, 19, true>
    {
        typedef SigmoidValuesTableQ45_19 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_45_19

    #if TINYMIND_USE_SIGMOID_46_18
    template<>
    struct SigmoidTableValueSize<46, 18, true>
    {
        typedef SigmoidValuesTableQ46_18 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_46_18

    #if TINYMIND_USE_SIGMOID_47_17
    template<>
    struct SigmoidTableValueSize<47, 17, true>
    {
        typedef SigmoidValuesTableQ47_17 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_47_17

    #if TINYMIND_USE_SIGMOID_48_16
    template<>
    struct SigmoidTableValueSize<48, 16, true>
    {
        typedef SigmoidValuesTableQ48_16 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_48_16

    #if TINYMIND_USE_SIGMOID_49_15
    template<>
    struct SigmoidTableValueSize<49, 15, true>
    {
        typedef SigmoidValuesTableQ49_15 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_49_15

    #if TINYMIND_USE_SIGMOID_50_14
    template<>
    struct SigmoidTableValueSize<50, 14, true>
    {
        typedef SigmoidValuesTableQ50_14 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_50_14

    #if TINYMIND_USE_SIGMOID_51_13
    template<>
    struct SigmoidTableValueSize<51, 13, true>
    {
        typedef SigmoidValuesTableQ51_13 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_51_13

    #if TINYMIND_USE_SIGMOID_52_12
    template<>
    struct SigmoidTableValueSize<52, 12, true>
    {
        typedef SigmoidValuesTableQ52_12 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_52_12

    #if TINYMIND_USE_SIGMOID_53_11
    template<>
    struct SigmoidTableValueSize<53, 11, true>
    {
        typedef SigmoidValuesTableQ53_11 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_53_11

    #if TINYMIND_USE_SIGMOID_54_10
    template<>
    struct SigmoidTableValueSize<54, 10, true>
    {
        typedef SigmoidValuesTableQ54_10 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_54_10

    #if TINYMIND_USE_SIGMOID_55_9
    template<>
    struct SigmoidTableValueSize<55, 9, true>
    {
        typedef SigmoidValuesTableQ55_9 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_55_9

    #if TINYMIND_USE_SIGMOID_56_8
    template<>
    struct SigmoidTableValueSize<56, 8, true>
    {
        typedef SigmoidValuesTableQ56_8 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_56_8

    #if TINYMIND_USE_SIGMOID_57_7
    template<>
    struct SigmoidTableValueSize<57, 7, true>
    {
        typedef SigmoidValuesTableQ57_7 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_57_7

    #if TINYMIND_USE_SIGMOID_58_6
    template<>
    struct SigmoidTableValueSize<58, 6, true>
    {
        typedef SigmoidValuesTableQ58_6 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_58_6

    #if TINYMIND_USE_SIGMOID_59_5
    template<>
    struct SigmoidTableValueSize<59, 5, true>
    {
        typedef SigmoidValuesTableQ59_5 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_59_5

    #if TINYMIND_USE_SIGMOID_60_4
    template<>
    struct SigmoidTableValueSize<60, 4, true>
    {
        typedef SigmoidValuesTableQ60_4 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_60_4

    #if TINYMIND_USE_SIGMOID_61_3
    template<>
    struct SigmoidTableValueSize<61, 3, true>
    {
        typedef SigmoidValuesTableQ61_3 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_61_3

    #if TINYMIND_USE_SIGMOID_62_2
    template<>
    struct SigmoidTableValueSize<62, 2, true>
    {
        typedef SigmoidValuesTableQ62_2 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_62_2

    #if TINYMIND_USE_SIGMOID_63_1
    template<>
    struct SigmoidTableValueSize<63, 1, true>
    {
        typedef SigmoidValuesTableQ63_1 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_63_1

    #if TINYMIND_USE_SIGMOID_1_127
    template<>
    struct SigmoidTableValueSize<1, 127, true>
    {
        typedef SigmoidValuesTableQ1_127 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_1_127

    #if TINYMIND_USE_SIGMOID_2_126
    template<>
    struct SigmoidTableValueSize<2, 126, true>
    {
        typedef SigmoidValuesTableQ2_126 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_2_126

    #if TINYMIND_USE_SIGMOID_3_125
    template<>
    struct SigmoidTableValueSize<3, 125, true>
    {
        typedef SigmoidValuesTableQ3_125 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_3_125

    #if TINYMIND_USE_SIGMOID_4_124
    template<>
    struct SigmoidTableValueSize<4, 124, true>
    {
        typedef SigmoidValuesTableQ4_124 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_4_124

    #if TINYMIND_USE_SIGMOID_5_123
    template<>
    struct SigmoidTableValueSize<5, 123, true>
    {
        typedef SigmoidValuesTableQ5_123 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_5_123

    #if TINYMIND_USE_SIGMOID_6_122
    template<>
    struct SigmoidTableValueSize<6, 122, true>
    {
        typedef SigmoidValuesTableQ6_122 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_6_122

    #if TINYMIND_USE_SIGMOID_7_121
    template<>
    struct SigmoidTableValueSize<7, 121, true>
    {
        typedef SigmoidValuesTableQ7_121 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_7_121

    #if TINYMIND_USE_SIGMOID_8_120
    template<>
    struct SigmoidTableValueSize<8, 120, true>
    {
        typedef SigmoidValuesTableQ8_120 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_8_120

    #if TINYMIND_USE_SIGMOID_9_119
    template<>
    struct SigmoidTableValueSize<9, 119, true>
    {
        typedef SigmoidValuesTableQ9_119 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_9_119

    #if TINYMIND_USE_SIGMOID_10_118
    template<>
    struct SigmoidTableValueSize<10, 118, true>
    {
        typedef SigmoidValuesTableQ10_118 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_10_118

    #if TINYMIND_USE_SIGMOID_11_117
    template<>
    struct SigmoidTableValueSize<11, 117, true>
    {
        typedef SigmoidValuesTableQ11_117 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_11_117

    #if TINYMIND_USE_SIGMOID_12_116
    template<>
    struct SigmoidTableValueSize<12, 116, true>
    {
        typedef SigmoidValuesTableQ12_116 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_12_116

    #if TINYMIND_USE_SIGMOID_13_115
    template<>
    struct SigmoidTableValueSize<13, 115, true>
    {
        typedef SigmoidValuesTableQ13_115 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_13_115

    #if TINYMIND_USE_SIGMOID_14_114
    template<>
    struct SigmoidTableValueSize<14, 114, true>
    {
        typedef SigmoidValuesTableQ14_114 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_14_114

    #if TINYMIND_USE_SIGMOID_15_113
    template<>
    struct SigmoidTableValueSize<15, 113, true>
    {
        typedef SigmoidValuesTableQ15_113 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_15_113

    #if TINYMIND_USE_SIGMOID_16_112
    template<>
    struct SigmoidTableValueSize<16, 112, true>
    {
        typedef SigmoidValuesTableQ16_112 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_16_112

    #if TINYMIND_USE_SIGMOID_17_111
    template<>
    struct SigmoidTableValueSize<17, 111, true>
    {
        typedef SigmoidValuesTableQ17_111 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_17_111

    #if TINYMIND_USE_SIGMOID_18_110
    template<>
    struct SigmoidTableValueSize<18, 110, true>
    {
        typedef SigmoidValuesTableQ18_110 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_18_110

    #if TINYMIND_USE_SIGMOID_19_109
    template<>
    struct SigmoidTableValueSize<19, 109, true>
    {
        typedef SigmoidValuesTableQ19_109 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_19_109

    #if TINYMIND_USE_SIGMOID_20_108
    template<>
    struct SigmoidTableValueSize<20, 108, true>
    {
        typedef SigmoidValuesTableQ20_108 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_20_108

    #if TINYMIND_USE_SIGMOID_21_107
    template<>
    struct SigmoidTableValueSize<21, 107, true>
    {
        typedef SigmoidValuesTableQ21_107 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_21_107

    #if TINYMIND_USE_SIGMOID_22_106
    template<>
    struct SigmoidTableValueSize<22, 106, true>
    {
        typedef SigmoidValuesTableQ22_106 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_22_106

    #if TINYMIND_USE_SIGMOID_23_105
    template<>
    struct SigmoidTableValueSize<23, 105, true>
    {
        typedef SigmoidValuesTableQ23_105 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_23_105

    #if TINYMIND_USE_SIGMOID_24_104
    template<>
    struct SigmoidTableValueSize<24, 104, true>
    {
        typedef SigmoidValuesTableQ24_104 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_24_104

    #if TINYMIND_USE_SIGMOID_25_103
    template<>
    struct SigmoidTableValueSize<25, 103, true>
    {
        typedef SigmoidValuesTableQ25_103 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_25_103

    #if TINYMIND_USE_SIGMOID_26_102
    template<>
    struct SigmoidTableValueSize<26, 102, true>
    {
        typedef SigmoidValuesTableQ26_102 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_26_102

    #if TINYMIND_USE_SIGMOID_27_101
    template<>
    struct SigmoidTableValueSize<27, 101, true>
    {
        typedef SigmoidValuesTableQ27_101 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_27_101

    #if TINYMIND_USE_SIGMOID_28_100
    template<>
    struct SigmoidTableValueSize<28, 100, true>
    {
        typedef SigmoidValuesTableQ28_100 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_28_100

    #if TINYMIND_USE_SIGMOID_29_99
    template<>
    struct SigmoidTableValueSize<29, 99, true>
    {
        typedef SigmoidValuesTableQ29_99 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_29_99

    #if TINYMIND_USE_SIGMOID_30_98
    template<>
    struct SigmoidTableValueSize<30, 98, true>
    {
        typedef SigmoidValuesTableQ30_98 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_30_98

    #if TINYMIND_USE_SIGMOID_31_97
    template<>
    struct SigmoidTableValueSize<31, 97, true>
    {
        typedef SigmoidValuesTableQ31_97 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_31_97

    #if TINYMIND_USE_SIGMOID_32_96
    template<>
    struct SigmoidTableValueSize<32, 96, true>
    {
        typedef SigmoidValuesTableQ32_96 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_32_96

    #if TINYMIND_USE_SIGMOID_33_95
    template<>
    struct SigmoidTableValueSize<33, 95, true>
    {
        typedef SigmoidValuesTableQ33_95 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_33_95

    #if TINYMIND_USE_SIGMOID_34_94
    template<>
    struct SigmoidTableValueSize<34, 94, true>
    {
        typedef SigmoidValuesTableQ34_94 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_34_94

    #if TINYMIND_USE_SIGMOID_35_93
    template<>
    struct SigmoidTableValueSize<35, 93, true>
    {
        typedef SigmoidValuesTableQ35_93 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_35_93

    #if TINYMIND_USE_SIGMOID_36_92
    template<>
    struct SigmoidTableValueSize<36, 92, true>
    {
        typedef SigmoidValuesTableQ36_92 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_36_92

    #if TINYMIND_USE_SIGMOID_37_91
    template<>
    struct SigmoidTableValueSize<37, 91, true>
    {
        typedef SigmoidValuesTableQ37_91 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_37_91

    #if TINYMIND_USE_SIGMOID_38_90
    template<>
    struct SigmoidTableValueSize<38, 90, true>
    {
        typedef SigmoidValuesTableQ38_90 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_38_90

    #if TINYMIND_USE_SIGMOID_39_89
    template<>
    struct SigmoidTableValueSize<39, 89, true>
    {
        typedef SigmoidValuesTableQ39_89 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_39_89

    #if TINYMIND_USE_SIGMOID_40_88
    template<>
    struct SigmoidTableValueSize<40, 88, true>
    {
        typedef SigmoidValuesTableQ40_88 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_40_88

    #if TINYMIND_USE_SIGMOID_41_87
    template<>
    struct SigmoidTableValueSize<41, 87, true>
    {
        typedef SigmoidValuesTableQ41_87 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_41_87

    #if TINYMIND_USE_SIGMOID_42_86
    template<>
    struct SigmoidTableValueSize<42, 86, true>
    {
        typedef SigmoidValuesTableQ42_86 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_42_86

    #if TINYMIND_USE_SIGMOID_43_85
    template<>
    struct SigmoidTableValueSize<43, 85, true>
    {
        typedef SigmoidValuesTableQ43_85 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_43_85

    #if TINYMIND_USE_SIGMOID_44_84
    template<>
    struct SigmoidTableValueSize<44, 84, true>
    {
        typedef SigmoidValuesTableQ44_84 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_44_84

    #if TINYMIND_USE_SIGMOID_45_83
    template<>
    struct SigmoidTableValueSize<45, 83, true>
    {
        typedef SigmoidValuesTableQ45_83 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_45_83

    #if TINYMIND_USE_SIGMOID_46_82
    template<>
    struct SigmoidTableValueSize<46, 82, true>
    {
        typedef SigmoidValuesTableQ46_82 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_46_82

    #if TINYMIND_USE_SIGMOID_47_81
    template<>
    struct SigmoidTableValueSize<47, 81, true>
    {
        typedef SigmoidValuesTableQ47_81 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_47_81

    #if TINYMIND_USE_SIGMOID_48_80
    template<>
    struct SigmoidTableValueSize<48, 80, true>
    {
        typedef SigmoidValuesTableQ48_80 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_48_80

    #if TINYMIND_USE_SIGMOID_49_79
    template<>
    struct SigmoidTableValueSize<49, 79, true>
    {
        typedef SigmoidValuesTableQ49_79 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_49_79

    #if TINYMIND_USE_SIGMOID_50_78
    template<>
    struct SigmoidTableValueSize<50, 78, true>
    {
        typedef SigmoidValuesTableQ50_78 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_50_78

    #if TINYMIND_USE_SIGMOID_51_77
    template<>
    struct SigmoidTableValueSize<51, 77, true>
    {
        typedef SigmoidValuesTableQ51_77 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_51_77

    #if TINYMIND_USE_SIGMOID_52_76
    template<>
    struct SigmoidTableValueSize<52, 76, true>
    {
        typedef SigmoidValuesTableQ52_76 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_52_76

    #if TINYMIND_USE_SIGMOID_53_75
    template<>
    struct SigmoidTableValueSize<53, 75, true>
    {
        typedef SigmoidValuesTableQ53_75 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_53_75

    #if TINYMIND_USE_SIGMOID_54_74
    template<>
    struct SigmoidTableValueSize<54, 74, true>
    {
        typedef SigmoidValuesTableQ54_74 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_54_74

    #if TINYMIND_USE_SIGMOID_55_73
    template<>
    struct SigmoidTableValueSize<55, 73, true>
    {
        typedef SigmoidValuesTableQ55_73 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_55_73

    #if TINYMIND_USE_SIGMOID_56_72
    template<>
    struct SigmoidTableValueSize<56, 72, true>
    {
        typedef SigmoidValuesTableQ56_72 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_56_72

    #if TINYMIND_USE_SIGMOID_57_71
    template<>
    struct SigmoidTableValueSize<57, 71, true>
    {
        typedef SigmoidValuesTableQ57_71 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_57_71

    #if TINYMIND_USE_SIGMOID_58_70
    template<>
    struct SigmoidTableValueSize<58, 70, true>
    {
        typedef SigmoidValuesTableQ58_70 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_58_70

    #if TINYMIND_USE_SIGMOID_59_69
    template<>
    struct SigmoidTableValueSize<59, 69, true>
    {
        typedef SigmoidValuesTableQ59_69 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_59_69

    #if TINYMIND_USE_SIGMOID_60_68
    template<>
    struct SigmoidTableValueSize<60, 68, true>
    {
        typedef SigmoidValuesTableQ60_68 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_60_68

    #if TINYMIND_USE_SIGMOID_61_67
    template<>
    struct SigmoidTableValueSize<61, 67, true>
    {
        typedef SigmoidValuesTableQ61_67 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_61_67

    #if TINYMIND_USE_SIGMOID_62_66
    template<>
    struct SigmoidTableValueSize<62, 66, true>
    {
        typedef SigmoidValuesTableQ62_66 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_62_66

    #if TINYMIND_USE_SIGMOID_63_65
    template<>
    struct SigmoidTableValueSize<63, 65, true>
    {
        typedef SigmoidValuesTableQ63_65 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_63_65

    #if TINYMIND_USE_SIGMOID_64_64
    template<>
    struct SigmoidTableValueSize<64, 64, true>
    {
        typedef SigmoidValuesTableQ64_64 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_64_64

    #if TINYMIND_USE_SIGMOID_65_63
    template<>
    struct SigmoidTableValueSize<65, 63, true>
    {
        typedef SigmoidValuesTableQ65_63 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_65_63

    #if TINYMIND_USE_SIGMOID_66_62
    template<>
    struct SigmoidTableValueSize<66, 62, true>
    {
        typedef SigmoidValuesTableQ66_62 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_66_62

    #if TINYMIND_USE_SIGMOID_67_61
    template<>
    struct SigmoidTableValueSize<67, 61, true>
    {
        typedef SigmoidValuesTableQ67_61 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_67_61

    #if TINYMIND_USE_SIGMOID_68_60
    template<>
    struct SigmoidTableValueSize<68, 60, true>
    {
        typedef SigmoidValuesTableQ68_60 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_68_60

    #if TINYMIND_USE_SIGMOID_69_59
    template<>
    struct SigmoidTableValueSize<69, 59, true>
    {
        typedef SigmoidValuesTableQ69_59 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_69_59

    #if TINYMIND_USE_SIGMOID_70_58
    template<>
    struct SigmoidTableValueSize<70, 58, true>
    {
        typedef SigmoidValuesTableQ70_58 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_70_58

    #if TINYMIND_USE_SIGMOID_71_57
    template<>
    struct SigmoidTableValueSize<71, 57, true>
    {
        typedef SigmoidValuesTableQ71_57 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_71_57

    #if TINYMIND_USE_SIGMOID_72_56
    template<>
    struct SigmoidTableValueSize<72, 56, true>
    {
        typedef SigmoidValuesTableQ72_56 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_72_56

    #if TINYMIND_USE_SIGMOID_73_55
    template<>
    struct SigmoidTableValueSize<73, 55, true>
    {
        typedef SigmoidValuesTableQ73_55 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_73_55

    #if TINYMIND_USE_SIGMOID_74_54
    template<>
    struct SigmoidTableValueSize<74, 54, true>
    {
        typedef SigmoidValuesTableQ74_54 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_74_54

    #if TINYMIND_USE_SIGMOID_75_53
    template<>
    struct SigmoidTableValueSize<75, 53, true>
    {
        typedef SigmoidValuesTableQ75_53 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_75_53

    #if TINYMIND_USE_SIGMOID_76_52
    template<>
    struct SigmoidTableValueSize<76, 52, true>
    {
        typedef SigmoidValuesTableQ76_52 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_76_52

    #if TINYMIND_USE_SIGMOID_77_51
    template<>
    struct SigmoidTableValueSize<77, 51, true>
    {
        typedef SigmoidValuesTableQ77_51 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_77_51

    #if TINYMIND_USE_SIGMOID_78_50
    template<>
    struct SigmoidTableValueSize<78, 50, true>
    {
        typedef SigmoidValuesTableQ78_50 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_78_50

    #if TINYMIND_USE_SIGMOID_79_49
    template<>
    struct SigmoidTableValueSize<79, 49, true>
    {
        typedef SigmoidValuesTableQ79_49 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_79_49

    #if TINYMIND_USE_SIGMOID_80_48
    template<>
    struct SigmoidTableValueSize<80, 48, true>
    {
        typedef SigmoidValuesTableQ80_48 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_80_48

    #if TINYMIND_USE_SIGMOID_81_47
    template<>
    struct SigmoidTableValueSize<81, 47, true>
    {
        typedef SigmoidValuesTableQ81_47 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_81_47

    #if TINYMIND_USE_SIGMOID_82_46
    template<>
    struct SigmoidTableValueSize<82, 46, true>
    {
        typedef SigmoidValuesTableQ82_46 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_82_46

    #if TINYMIND_USE_SIGMOID_83_45
    template<>
    struct SigmoidTableValueSize<83, 45, true>
    {
        typedef SigmoidValuesTableQ83_45 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_83_45

    #if TINYMIND_USE_SIGMOID_84_44
    template<>
    struct SigmoidTableValueSize<84, 44, true>
    {
        typedef SigmoidValuesTableQ84_44 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_84_44

    #if TINYMIND_USE_SIGMOID_85_43
    template<>
    struct SigmoidTableValueSize<85, 43, true>
    {
        typedef SigmoidValuesTableQ85_43 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_85_43

    #if TINYMIND_USE_SIGMOID_86_42
    template<>
    struct SigmoidTableValueSize<86, 42, true>
    {
        typedef SigmoidValuesTableQ86_42 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_86_42

    #if TINYMIND_USE_SIGMOID_87_41
    template<>
    struct SigmoidTableValueSize<87, 41, true>
    {
        typedef SigmoidValuesTableQ87_41 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_87_41

    #if TINYMIND_USE_SIGMOID_88_40
    template<>
    struct SigmoidTableValueSize<88, 40, true>
    {
        typedef SigmoidValuesTableQ88_40 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_88_40

    #if TINYMIND_USE_SIGMOID_89_39
    template<>
    struct SigmoidTableValueSize<89, 39, true>
    {
        typedef SigmoidValuesTableQ89_39 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_89_39

    #if TINYMIND_USE_SIGMOID_90_38
    template<>
    struct SigmoidTableValueSize<90, 38, true>
    {
        typedef SigmoidValuesTableQ90_38 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_90_38

    #if TINYMIND_USE_SIGMOID_91_37
    template<>
    struct SigmoidTableValueSize<91, 37, true>
    {
        typedef SigmoidValuesTableQ91_37 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_91_37

    #if TINYMIND_USE_SIGMOID_92_36
    template<>
    struct SigmoidTableValueSize<92, 36, true>
    {
        typedef SigmoidValuesTableQ92_36 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_92_36

    #if TINYMIND_USE_SIGMOID_93_35
    template<>
    struct SigmoidTableValueSize<93, 35, true>
    {
        typedef SigmoidValuesTableQ93_35 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_93_35

    #if TINYMIND_USE_SIGMOID_94_34
    template<>
    struct SigmoidTableValueSize<94, 34, true>
    {
        typedef SigmoidValuesTableQ94_34 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_94_34

    #if TINYMIND_USE_SIGMOID_95_33
    template<>
    struct SigmoidTableValueSize<95, 33, true>
    {
        typedef SigmoidValuesTableQ95_33 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_95_33

    #if TINYMIND_USE_SIGMOID_96_32
    template<>
    struct SigmoidTableValueSize<96, 32, true>
    {
        typedef SigmoidValuesTableQ96_32 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_96_32

    #if TINYMIND_USE_SIGMOID_97_31
    template<>
    struct SigmoidTableValueSize<97, 31, true>
    {
        typedef SigmoidValuesTableQ97_31 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_97_31

    #if TINYMIND_USE_SIGMOID_98_30
    template<>
    struct SigmoidTableValueSize<98, 30, true>
    {
        typedef SigmoidValuesTableQ98_30 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_98_30

    #if TINYMIND_USE_SIGMOID_99_29
    template<>
    struct SigmoidTableValueSize<99, 29, true>
    {
        typedef SigmoidValuesTableQ99_29 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_99_29

    #if TINYMIND_USE_SIGMOID_100_28
    template<>
    struct SigmoidTableValueSize<100, 28, true>
    {
        typedef SigmoidValuesTableQ100_28 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_100_28

    #if TINYMIND_USE_SIGMOID_101_27
    template<>
    struct SigmoidTableValueSize<101, 27, true>
    {
        typedef SigmoidValuesTableQ101_27 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_101_27

    #if TINYMIND_USE_SIGMOID_102_26
    template<>
    struct SigmoidTableValueSize<102, 26, true>
    {
        typedef SigmoidValuesTableQ102_26 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_102_26

    #if TINYMIND_USE_SIGMOID_103_25
    template<>
    struct SigmoidTableValueSize<103, 25, true>
    {
        typedef SigmoidValuesTableQ103_25 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_103_25

    #if TINYMIND_USE_SIGMOID_104_24
    template<>
    struct SigmoidTableValueSize<104, 24, true>
    {
        typedef SigmoidValuesTableQ104_24 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_104_24

    #if TINYMIND_USE_SIGMOID_105_23
    template<>
    struct SigmoidTableValueSize<105, 23, true>
    {
        typedef SigmoidValuesTableQ105_23 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_105_23

    #if TINYMIND_USE_SIGMOID_106_22
    template<>
    struct SigmoidTableValueSize<106, 22, true>
    {
        typedef SigmoidValuesTableQ106_22 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_106_22

    #if TINYMIND_USE_SIGMOID_107_21
    template<>
    struct SigmoidTableValueSize<107, 21, true>
    {
        typedef SigmoidValuesTableQ107_21 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_107_21

    #if TINYMIND_USE_SIGMOID_108_20
    template<>
    struct SigmoidTableValueSize<108, 20, true>
    {
        typedef SigmoidValuesTableQ108_20 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_108_20

    #if TINYMIND_USE_SIGMOID_109_19
    template<>
    struct SigmoidTableValueSize<109, 19, true>
    {
        typedef SigmoidValuesTableQ109_19 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_109_19

    #if TINYMIND_USE_SIGMOID_110_18
    template<>
    struct SigmoidTableValueSize<110, 18, true>
    {
        typedef SigmoidValuesTableQ110_18 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_110_18

    #if TINYMIND_USE_SIGMOID_111_17
    template<>
    struct SigmoidTableValueSize<111, 17, true>
    {
        typedef SigmoidValuesTableQ111_17 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_111_17

    #if TINYMIND_USE_SIGMOID_112_16
    template<>
    struct SigmoidTableValueSize<112, 16, true>
    {
        typedef SigmoidValuesTableQ112_16 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_112_16

    #if TINYMIND_USE_SIGMOID_113_15
    template<>
    struct SigmoidTableValueSize<113, 15, true>
    {
        typedef SigmoidValuesTableQ113_15 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_113_15

    #if TINYMIND_USE_SIGMOID_114_14
    template<>
    struct SigmoidTableValueSize<114, 14, true>
    {
        typedef SigmoidValuesTableQ114_14 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_114_14

    #if TINYMIND_USE_SIGMOID_115_13
    template<>
    struct SigmoidTableValueSize<115, 13, true>
    {
        typedef SigmoidValuesTableQ115_13 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_115_13

    #if TINYMIND_USE_SIGMOID_116_12
    template<>
    struct SigmoidTableValueSize<116, 12, true>
    {
        typedef SigmoidValuesTableQ116_12 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_116_12

    #if TINYMIND_USE_SIGMOID_117_11
    template<>
    struct SigmoidTableValueSize<117, 11, true>
    {
        typedef SigmoidValuesTableQ117_11 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_117_11

    #if TINYMIND_USE_SIGMOID_118_10
    template<>
    struct SigmoidTableValueSize<118, 10, true>
    {
        typedef SigmoidValuesTableQ118_10 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_118_10

    #if TINYMIND_USE_SIGMOID_119_9
    template<>
    struct SigmoidTableValueSize<119, 9, true>
    {
        typedef SigmoidValuesTableQ119_9 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_119_9

    #if TINYMIND_USE_SIGMOID_120_8
    template<>
    struct SigmoidTableValueSize<120, 8, true>
    {
        typedef SigmoidValuesTableQ120_8 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_120_8

    #if TINYMIND_USE_SIGMOID_121_7
    template<>
    struct SigmoidTableValueSize<121, 7, true>
    {
        typedef SigmoidValuesTableQ121_7 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_121_7

    #if TINYMIND_USE_SIGMOID_122_6
    template<>
    struct SigmoidTableValueSize<122, 6, true>
    {
        typedef SigmoidValuesTableQ122_6 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_122_6

    #if TINYMIND_USE_SIGMOID_123_5
    template<>
    struct SigmoidTableValueSize<123, 5, true>
    {
        typedef SigmoidValuesTableQ123_5 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_123_5

    #if TINYMIND_USE_SIGMOID_124_4
    template<>
    struct SigmoidTableValueSize<124, 4, true>
    {
        typedef SigmoidValuesTableQ124_4 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_124_4

    #if TINYMIND_USE_SIGMOID_125_3
    template<>
    struct SigmoidTableValueSize<125, 3, true>
    {
        typedef SigmoidValuesTableQ125_3 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_125_3

    #if TINYMIND_USE_SIGMOID_126_2
    template<>
    struct SigmoidTableValueSize<126, 2, true>
    {
        typedef SigmoidValuesTableQ126_2 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_126_2

    #if TINYMIND_USE_SIGMOID_127_1
    template<>
    struct SigmoidTableValueSize<127, 1, true>
    {
        typedef SigmoidValuesTableQ127_1 SigmoidTableType;
    };
    #endif // TINYMIND_USE_SIGMOID_127_1

    template<unsigned FixedBits,unsigned FracBits, bool IsSigned>
    struct SigmoidValuesTableSelector
    {
        typedef typename SigmoidTableValueSize<FixedBits, FracBits, IsSigned>::SigmoidTableType SigmoidTableType;
    };
}
