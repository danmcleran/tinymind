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

#include "expValues8Bit.hpp"
#include "expValues16Bit.hpp"
#include "expValues32Bit.hpp"
#include "expValues64Bit.hpp"
#include "expValues128Bit.hpp"

namespace tinymind {
    template<unsigned FixedBits, unsigned FracBits, bool IsSigned>
    struct ExpTableValueSize
    {
    };

    #if TINYMIND_USE_EXP_1_7
    template<>
    struct ExpTableValueSize<1, 7, true>
    {
        typedef ExpValuesTableQ1_7 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_1_7

    #if TINYMIND_USE_EXP_2_6
    template<>
    struct ExpTableValueSize<2, 6, true>
    {
        typedef ExpValuesTableQ2_6 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_2_6

    #if TINYMIND_USE_EXP_3_5
    template<>
    struct ExpTableValueSize<3, 5, true>
    {
        typedef ExpValuesTableQ3_5 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_3_5

    #if TINYMIND_USE_EXP_4_4
    template<>
    struct ExpTableValueSize<4, 4, true>
    {
        typedef ExpValuesTableQ4_4 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_4_4

    #if TINYMIND_USE_EXP_5_3
    template<>
    struct ExpTableValueSize<5, 3, true>
    {
        typedef ExpValuesTableQ5_3 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_5_3

    #if TINYMIND_USE_EXP_6_2
    template<>
    struct ExpTableValueSize<6, 2, true>
    {
        typedef ExpValuesTableQ6_2 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_6_2

    #if TINYMIND_USE_EXP_7_1
    template<>
    struct ExpTableValueSize<7, 1, true>
    {
        typedef ExpValuesTableQ7_1 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_7_1

    #if TINYMIND_USE_EXP_1_15
    template<>
    struct ExpTableValueSize<1, 15, true>
    {
        typedef ExpValuesTableQ1_15 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_1_15

    #if TINYMIND_USE_EXP_2_14
    template<>
    struct ExpTableValueSize<2, 14, true>
    {
        typedef ExpValuesTableQ2_14 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_2_14

    #if TINYMIND_USE_EXP_3_13
    template<>
    struct ExpTableValueSize<3, 13, true>
    {
        typedef ExpValuesTableQ3_13 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_3_13

    #if TINYMIND_USE_EXP_4_12
    template<>
    struct ExpTableValueSize<4, 12, true>
    {
        typedef ExpValuesTableQ4_12 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_4_12

    #if TINYMIND_USE_EXP_5_11
    template<>
    struct ExpTableValueSize<5, 11, true>
    {
        typedef ExpValuesTableQ5_11 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_5_11

    #if TINYMIND_USE_EXP_6_10
    template<>
    struct ExpTableValueSize<6, 10, true>
    {
        typedef ExpValuesTableQ6_10 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_6_10

    #if TINYMIND_USE_EXP_7_9
    template<>
    struct ExpTableValueSize<7, 9, true>
    {
        typedef ExpValuesTableQ7_9 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_7_9

    #if TINYMIND_USE_EXP_8_8
    template<>
    struct ExpTableValueSize<8, 8, true>
    {
        typedef ExpValuesTableQ8_8 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_8_8

    #if TINYMIND_USE_EXP_9_7
    template<>
    struct ExpTableValueSize<9, 7, true>
    {
        typedef ExpValuesTableQ9_7 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_9_7

    #if TINYMIND_USE_EXP_10_6
    template<>
    struct ExpTableValueSize<10, 6, true>
    {
        typedef ExpValuesTableQ10_6 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_10_6

    #if TINYMIND_USE_EXP_11_5
    template<>
    struct ExpTableValueSize<11, 5, true>
    {
        typedef ExpValuesTableQ11_5 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_11_5

    #if TINYMIND_USE_EXP_12_4
    template<>
    struct ExpTableValueSize<12, 4, true>
    {
        typedef ExpValuesTableQ12_4 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_12_4

    #if TINYMIND_USE_EXP_13_3
    template<>
    struct ExpTableValueSize<13, 3, true>
    {
        typedef ExpValuesTableQ13_3 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_13_3

    #if TINYMIND_USE_EXP_14_2
    template<>
    struct ExpTableValueSize<14, 2, true>
    {
        typedef ExpValuesTableQ14_2 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_14_2

    #if TINYMIND_USE_EXP_15_1
    template<>
    struct ExpTableValueSize<15, 1, true>
    {
        typedef ExpValuesTableQ15_1 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_15_1

    #if TINYMIND_USE_EXP_1_31
    template<>
    struct ExpTableValueSize<1, 31, true>
    {
        typedef ExpValuesTableQ1_31 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_1_31

    #if TINYMIND_USE_EXP_2_30
    template<>
    struct ExpTableValueSize<2, 30, true>
    {
        typedef ExpValuesTableQ2_30 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_2_30

    #if TINYMIND_USE_EXP_3_29
    template<>
    struct ExpTableValueSize<3, 29, true>
    {
        typedef ExpValuesTableQ3_29 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_3_29

    #if TINYMIND_USE_EXP_4_28
    template<>
    struct ExpTableValueSize<4, 28, true>
    {
        typedef ExpValuesTableQ4_28 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_4_28

    #if TINYMIND_USE_EXP_5_27
    template<>
    struct ExpTableValueSize<5, 27, true>
    {
        typedef ExpValuesTableQ5_27 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_5_27

    #if TINYMIND_USE_EXP_6_26
    template<>
    struct ExpTableValueSize<6, 26, true>
    {
        typedef ExpValuesTableQ6_26 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_6_26

    #if TINYMIND_USE_EXP_7_25
    template<>
    struct ExpTableValueSize<7, 25, true>
    {
        typedef ExpValuesTableQ7_25 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_7_25

    #if TINYMIND_USE_EXP_8_24
    template<>
    struct ExpTableValueSize<8, 24, true>
    {
        typedef ExpValuesTableQ8_24 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_8_24

    #if TINYMIND_USE_EXP_9_23
    template<>
    struct ExpTableValueSize<9, 23, true>
    {
        typedef ExpValuesTableQ9_23 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_9_23

    #if TINYMIND_USE_EXP_10_22
    template<>
    struct ExpTableValueSize<10, 22, true>
    {
        typedef ExpValuesTableQ10_22 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_10_22

    #if TINYMIND_USE_EXP_11_21
    template<>
    struct ExpTableValueSize<11, 21, true>
    {
        typedef ExpValuesTableQ11_21 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_11_21

    #if TINYMIND_USE_EXP_12_20
    template<>
    struct ExpTableValueSize<12, 20, true>
    {
        typedef ExpValuesTableQ12_20 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_12_20

    #if TINYMIND_USE_EXP_13_19
    template<>
    struct ExpTableValueSize<13, 19, true>
    {
        typedef ExpValuesTableQ13_19 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_13_19

    #if TINYMIND_USE_EXP_14_18
    template<>
    struct ExpTableValueSize<14, 18, true>
    {
        typedef ExpValuesTableQ14_18 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_14_18

    #if TINYMIND_USE_EXP_15_17
    template<>
    struct ExpTableValueSize<15, 17, true>
    {
        typedef ExpValuesTableQ15_17 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_15_17

    #if TINYMIND_USE_EXP_16_16
    template<>
    struct ExpTableValueSize<16, 16, true>
    {
        typedef ExpValuesTableQ16_16 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_16_16

    #if TINYMIND_USE_EXP_17_15
    template<>
    struct ExpTableValueSize<17, 15, true>
    {
        typedef ExpValuesTableQ17_15 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_17_15

    #if TINYMIND_USE_EXP_18_14
    template<>
    struct ExpTableValueSize<18, 14, true>
    {
        typedef ExpValuesTableQ18_14 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_18_14

    #if TINYMIND_USE_EXP_19_13
    template<>
    struct ExpTableValueSize<19, 13, true>
    {
        typedef ExpValuesTableQ19_13 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_19_13

    #if TINYMIND_USE_EXP_20_12
    template<>
    struct ExpTableValueSize<20, 12, true>
    {
        typedef ExpValuesTableQ20_12 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_20_12

    #if TINYMIND_USE_EXP_21_11
    template<>
    struct ExpTableValueSize<21, 11, true>
    {
        typedef ExpValuesTableQ21_11 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_21_11

    #if TINYMIND_USE_EXP_22_10
    template<>
    struct ExpTableValueSize<22, 10, true>
    {
        typedef ExpValuesTableQ22_10 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_22_10

    #if TINYMIND_USE_EXP_23_9
    template<>
    struct ExpTableValueSize<23, 9, true>
    {
        typedef ExpValuesTableQ23_9 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_23_9

    #if TINYMIND_USE_EXP_24_8
    template<>
    struct ExpTableValueSize<24, 8, true>
    {
        typedef ExpValuesTableQ24_8 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_24_8

    #if TINYMIND_USE_EXP_25_7
    template<>
    struct ExpTableValueSize<25, 7, true>
    {
        typedef ExpValuesTableQ25_7 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_25_7

    #if TINYMIND_USE_EXP_26_6
    template<>
    struct ExpTableValueSize<26, 6, true>
    {
        typedef ExpValuesTableQ26_6 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_26_6

    #if TINYMIND_USE_EXP_27_5
    template<>
    struct ExpTableValueSize<27, 5, true>
    {
        typedef ExpValuesTableQ27_5 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_27_5

    #if TINYMIND_USE_EXP_28_4
    template<>
    struct ExpTableValueSize<28, 4, true>
    {
        typedef ExpValuesTableQ28_4 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_28_4

    #if TINYMIND_USE_EXP_29_3
    template<>
    struct ExpTableValueSize<29, 3, true>
    {
        typedef ExpValuesTableQ29_3 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_29_3

    #if TINYMIND_USE_EXP_30_2
    template<>
    struct ExpTableValueSize<30, 2, true>
    {
        typedef ExpValuesTableQ30_2 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_30_2

    #if TINYMIND_USE_EXP_31_1
    template<>
    struct ExpTableValueSize<31, 1, true>
    {
        typedef ExpValuesTableQ31_1 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_31_1

    #if TINYMIND_USE_EXP_1_63
    template<>
    struct ExpTableValueSize<1, 63, true>
    {
        typedef ExpValuesTableQ1_63 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_1_63

    #if TINYMIND_USE_EXP_2_62
    template<>
    struct ExpTableValueSize<2, 62, true>
    {
        typedef ExpValuesTableQ2_62 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_2_62

    #if TINYMIND_USE_EXP_3_61
    template<>
    struct ExpTableValueSize<3, 61, true>
    {
        typedef ExpValuesTableQ3_61 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_3_61

    #if TINYMIND_USE_EXP_4_60
    template<>
    struct ExpTableValueSize<4, 60, true>
    {
        typedef ExpValuesTableQ4_60 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_4_60

    #if TINYMIND_USE_EXP_5_59
    template<>
    struct ExpTableValueSize<5, 59, true>
    {
        typedef ExpValuesTableQ5_59 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_5_59

    #if TINYMIND_USE_EXP_6_58
    template<>
    struct ExpTableValueSize<6, 58, true>
    {
        typedef ExpValuesTableQ6_58 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_6_58

    #if TINYMIND_USE_EXP_7_57
    template<>
    struct ExpTableValueSize<7, 57, true>
    {
        typedef ExpValuesTableQ7_57 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_7_57

    #if TINYMIND_USE_EXP_8_56
    template<>
    struct ExpTableValueSize<8, 56, true>
    {
        typedef ExpValuesTableQ8_56 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_8_56

    #if TINYMIND_USE_EXP_9_55
    template<>
    struct ExpTableValueSize<9, 55, true>
    {
        typedef ExpValuesTableQ9_55 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_9_55

    #if TINYMIND_USE_EXP_10_54
    template<>
    struct ExpTableValueSize<10, 54, true>
    {
        typedef ExpValuesTableQ10_54 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_10_54

    #if TINYMIND_USE_EXP_11_53
    template<>
    struct ExpTableValueSize<11, 53, true>
    {
        typedef ExpValuesTableQ11_53 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_11_53

    #if TINYMIND_USE_EXP_12_52
    template<>
    struct ExpTableValueSize<12, 52, true>
    {
        typedef ExpValuesTableQ12_52 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_12_52

    #if TINYMIND_USE_EXP_13_51
    template<>
    struct ExpTableValueSize<13, 51, true>
    {
        typedef ExpValuesTableQ13_51 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_13_51

    #if TINYMIND_USE_EXP_14_50
    template<>
    struct ExpTableValueSize<14, 50, true>
    {
        typedef ExpValuesTableQ14_50 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_14_50

    #if TINYMIND_USE_EXP_15_49
    template<>
    struct ExpTableValueSize<15, 49, true>
    {
        typedef ExpValuesTableQ15_49 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_15_49

    #if TINYMIND_USE_EXP_16_48
    template<>
    struct ExpTableValueSize<16, 48, true>
    {
        typedef ExpValuesTableQ16_48 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_16_48

    #if TINYMIND_USE_EXP_17_47
    template<>
    struct ExpTableValueSize<17, 47, true>
    {
        typedef ExpValuesTableQ17_47 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_17_47

    #if TINYMIND_USE_EXP_18_46
    template<>
    struct ExpTableValueSize<18, 46, true>
    {
        typedef ExpValuesTableQ18_46 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_18_46

    #if TINYMIND_USE_EXP_19_45
    template<>
    struct ExpTableValueSize<19, 45, true>
    {
        typedef ExpValuesTableQ19_45 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_19_45

    #if TINYMIND_USE_EXP_20_44
    template<>
    struct ExpTableValueSize<20, 44, true>
    {
        typedef ExpValuesTableQ20_44 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_20_44

    #if TINYMIND_USE_EXP_21_43
    template<>
    struct ExpTableValueSize<21, 43, true>
    {
        typedef ExpValuesTableQ21_43 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_21_43

    #if TINYMIND_USE_EXP_22_42
    template<>
    struct ExpTableValueSize<22, 42, true>
    {
        typedef ExpValuesTableQ22_42 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_22_42

    #if TINYMIND_USE_EXP_23_41
    template<>
    struct ExpTableValueSize<23, 41, true>
    {
        typedef ExpValuesTableQ23_41 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_23_41

    #if TINYMIND_USE_EXP_24_40
    template<>
    struct ExpTableValueSize<24, 40, true>
    {
        typedef ExpValuesTableQ24_40 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_24_40

    #if TINYMIND_USE_EXP_25_39
    template<>
    struct ExpTableValueSize<25, 39, true>
    {
        typedef ExpValuesTableQ25_39 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_25_39

    #if TINYMIND_USE_EXP_26_38
    template<>
    struct ExpTableValueSize<26, 38, true>
    {
        typedef ExpValuesTableQ26_38 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_26_38

    #if TINYMIND_USE_EXP_27_37
    template<>
    struct ExpTableValueSize<27, 37, true>
    {
        typedef ExpValuesTableQ27_37 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_27_37

    #if TINYMIND_USE_EXP_28_36
    template<>
    struct ExpTableValueSize<28, 36, true>
    {
        typedef ExpValuesTableQ28_36 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_28_36

    #if TINYMIND_USE_EXP_29_35
    template<>
    struct ExpTableValueSize<29, 35, true>
    {
        typedef ExpValuesTableQ29_35 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_29_35

    #if TINYMIND_USE_EXP_30_34
    template<>
    struct ExpTableValueSize<30, 34, true>
    {
        typedef ExpValuesTableQ30_34 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_30_34

    #if TINYMIND_USE_EXP_31_33
    template<>
    struct ExpTableValueSize<31, 33, true>
    {
        typedef ExpValuesTableQ31_33 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_31_33

    #if TINYMIND_USE_EXP_32_32
    template<>
    struct ExpTableValueSize<32, 32, true>
    {
        typedef ExpValuesTableQ32_32 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_32_32

    #if TINYMIND_USE_EXP_33_31
    template<>
    struct ExpTableValueSize<33, 31, true>
    {
        typedef ExpValuesTableQ33_31 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_33_31

    #if TINYMIND_USE_EXP_34_30
    template<>
    struct ExpTableValueSize<34, 30, true>
    {
        typedef ExpValuesTableQ34_30 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_34_30

    #if TINYMIND_USE_EXP_35_29
    template<>
    struct ExpTableValueSize<35, 29, true>
    {
        typedef ExpValuesTableQ35_29 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_35_29

    #if TINYMIND_USE_EXP_36_28
    template<>
    struct ExpTableValueSize<36, 28, true>
    {
        typedef ExpValuesTableQ36_28 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_36_28

    #if TINYMIND_USE_EXP_37_27
    template<>
    struct ExpTableValueSize<37, 27, true>
    {
        typedef ExpValuesTableQ37_27 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_37_27

    #if TINYMIND_USE_EXP_38_26
    template<>
    struct ExpTableValueSize<38, 26, true>
    {
        typedef ExpValuesTableQ38_26 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_38_26

    #if TINYMIND_USE_EXP_39_25
    template<>
    struct ExpTableValueSize<39, 25, true>
    {
        typedef ExpValuesTableQ39_25 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_39_25

    #if TINYMIND_USE_EXP_40_24
    template<>
    struct ExpTableValueSize<40, 24, true>
    {
        typedef ExpValuesTableQ40_24 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_40_24

    #if TINYMIND_USE_EXP_41_23
    template<>
    struct ExpTableValueSize<41, 23, true>
    {
        typedef ExpValuesTableQ41_23 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_41_23

    #if TINYMIND_USE_EXP_42_22
    template<>
    struct ExpTableValueSize<42, 22, true>
    {
        typedef ExpValuesTableQ42_22 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_42_22

    #if TINYMIND_USE_EXP_43_21
    template<>
    struct ExpTableValueSize<43, 21, true>
    {
        typedef ExpValuesTableQ43_21 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_43_21

    #if TINYMIND_USE_EXP_44_20
    template<>
    struct ExpTableValueSize<44, 20, true>
    {
        typedef ExpValuesTableQ44_20 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_44_20

    #if TINYMIND_USE_EXP_45_19
    template<>
    struct ExpTableValueSize<45, 19, true>
    {
        typedef ExpValuesTableQ45_19 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_45_19

    #if TINYMIND_USE_EXP_46_18
    template<>
    struct ExpTableValueSize<46, 18, true>
    {
        typedef ExpValuesTableQ46_18 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_46_18

    #if TINYMIND_USE_EXP_47_17
    template<>
    struct ExpTableValueSize<47, 17, true>
    {
        typedef ExpValuesTableQ47_17 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_47_17

    #if TINYMIND_USE_EXP_48_16
    template<>
    struct ExpTableValueSize<48, 16, true>
    {
        typedef ExpValuesTableQ48_16 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_48_16

    #if TINYMIND_USE_EXP_49_15
    template<>
    struct ExpTableValueSize<49, 15, true>
    {
        typedef ExpValuesTableQ49_15 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_49_15

    #if TINYMIND_USE_EXP_50_14
    template<>
    struct ExpTableValueSize<50, 14, true>
    {
        typedef ExpValuesTableQ50_14 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_50_14

    #if TINYMIND_USE_EXP_51_13
    template<>
    struct ExpTableValueSize<51, 13, true>
    {
        typedef ExpValuesTableQ51_13 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_51_13

    #if TINYMIND_USE_EXP_52_12
    template<>
    struct ExpTableValueSize<52, 12, true>
    {
        typedef ExpValuesTableQ52_12 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_52_12

    #if TINYMIND_USE_EXP_53_11
    template<>
    struct ExpTableValueSize<53, 11, true>
    {
        typedef ExpValuesTableQ53_11 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_53_11

    #if TINYMIND_USE_EXP_54_10
    template<>
    struct ExpTableValueSize<54, 10, true>
    {
        typedef ExpValuesTableQ54_10 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_54_10

    #if TINYMIND_USE_EXP_55_9
    template<>
    struct ExpTableValueSize<55, 9, true>
    {
        typedef ExpValuesTableQ55_9 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_55_9

    #if TINYMIND_USE_EXP_56_8
    template<>
    struct ExpTableValueSize<56, 8, true>
    {
        typedef ExpValuesTableQ56_8 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_56_8

    #if TINYMIND_USE_EXP_57_7
    template<>
    struct ExpTableValueSize<57, 7, true>
    {
        typedef ExpValuesTableQ57_7 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_57_7

    #if TINYMIND_USE_EXP_58_6
    template<>
    struct ExpTableValueSize<58, 6, true>
    {
        typedef ExpValuesTableQ58_6 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_58_6

    #if TINYMIND_USE_EXP_59_5
    template<>
    struct ExpTableValueSize<59, 5, true>
    {
        typedef ExpValuesTableQ59_5 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_59_5

    #if TINYMIND_USE_EXP_60_4
    template<>
    struct ExpTableValueSize<60, 4, true>
    {
        typedef ExpValuesTableQ60_4 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_60_4

    #if TINYMIND_USE_EXP_61_3
    template<>
    struct ExpTableValueSize<61, 3, true>
    {
        typedef ExpValuesTableQ61_3 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_61_3

    #if TINYMIND_USE_EXP_62_2
    template<>
    struct ExpTableValueSize<62, 2, true>
    {
        typedef ExpValuesTableQ62_2 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_62_2

    #if TINYMIND_USE_EXP_63_1
    template<>
    struct ExpTableValueSize<63, 1, true>
    {
        typedef ExpValuesTableQ63_1 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_63_1

    #if TINYMIND_USE_EXP_1_127
    template<>
    struct ExpTableValueSize<1, 127, true>
    {
        typedef ExpValuesTableQ1_127 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_1_127

    #if TINYMIND_USE_EXP_2_126
    template<>
    struct ExpTableValueSize<2, 126, true>
    {
        typedef ExpValuesTableQ2_126 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_2_126

    #if TINYMIND_USE_EXP_3_125
    template<>
    struct ExpTableValueSize<3, 125, true>
    {
        typedef ExpValuesTableQ3_125 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_3_125

    #if TINYMIND_USE_EXP_4_124
    template<>
    struct ExpTableValueSize<4, 124, true>
    {
        typedef ExpValuesTableQ4_124 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_4_124

    #if TINYMIND_USE_EXP_5_123
    template<>
    struct ExpTableValueSize<5, 123, true>
    {
        typedef ExpValuesTableQ5_123 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_5_123

    #if TINYMIND_USE_EXP_6_122
    template<>
    struct ExpTableValueSize<6, 122, true>
    {
        typedef ExpValuesTableQ6_122 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_6_122

    #if TINYMIND_USE_EXP_7_121
    template<>
    struct ExpTableValueSize<7, 121, true>
    {
        typedef ExpValuesTableQ7_121 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_7_121

    #if TINYMIND_USE_EXP_8_120
    template<>
    struct ExpTableValueSize<8, 120, true>
    {
        typedef ExpValuesTableQ8_120 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_8_120

    #if TINYMIND_USE_EXP_9_119
    template<>
    struct ExpTableValueSize<9, 119, true>
    {
        typedef ExpValuesTableQ9_119 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_9_119

    #if TINYMIND_USE_EXP_10_118
    template<>
    struct ExpTableValueSize<10, 118, true>
    {
        typedef ExpValuesTableQ10_118 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_10_118

    #if TINYMIND_USE_EXP_11_117
    template<>
    struct ExpTableValueSize<11, 117, true>
    {
        typedef ExpValuesTableQ11_117 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_11_117

    #if TINYMIND_USE_EXP_12_116
    template<>
    struct ExpTableValueSize<12, 116, true>
    {
        typedef ExpValuesTableQ12_116 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_12_116

    #if TINYMIND_USE_EXP_13_115
    template<>
    struct ExpTableValueSize<13, 115, true>
    {
        typedef ExpValuesTableQ13_115 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_13_115

    #if TINYMIND_USE_EXP_14_114
    template<>
    struct ExpTableValueSize<14, 114, true>
    {
        typedef ExpValuesTableQ14_114 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_14_114

    #if TINYMIND_USE_EXP_15_113
    template<>
    struct ExpTableValueSize<15, 113, true>
    {
        typedef ExpValuesTableQ15_113 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_15_113

    #if TINYMIND_USE_EXP_16_112
    template<>
    struct ExpTableValueSize<16, 112, true>
    {
        typedef ExpValuesTableQ16_112 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_16_112

    #if TINYMIND_USE_EXP_17_111
    template<>
    struct ExpTableValueSize<17, 111, true>
    {
        typedef ExpValuesTableQ17_111 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_17_111

    #if TINYMIND_USE_EXP_18_110
    template<>
    struct ExpTableValueSize<18, 110, true>
    {
        typedef ExpValuesTableQ18_110 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_18_110

    #if TINYMIND_USE_EXP_19_109
    template<>
    struct ExpTableValueSize<19, 109, true>
    {
        typedef ExpValuesTableQ19_109 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_19_109

    #if TINYMIND_USE_EXP_20_108
    template<>
    struct ExpTableValueSize<20, 108, true>
    {
        typedef ExpValuesTableQ20_108 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_20_108

    #if TINYMIND_USE_EXP_21_107
    template<>
    struct ExpTableValueSize<21, 107, true>
    {
        typedef ExpValuesTableQ21_107 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_21_107

    #if TINYMIND_USE_EXP_22_106
    template<>
    struct ExpTableValueSize<22, 106, true>
    {
        typedef ExpValuesTableQ22_106 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_22_106

    #if TINYMIND_USE_EXP_23_105
    template<>
    struct ExpTableValueSize<23, 105, true>
    {
        typedef ExpValuesTableQ23_105 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_23_105

    #if TINYMIND_USE_EXP_24_104
    template<>
    struct ExpTableValueSize<24, 104, true>
    {
        typedef ExpValuesTableQ24_104 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_24_104

    #if TINYMIND_USE_EXP_25_103
    template<>
    struct ExpTableValueSize<25, 103, true>
    {
        typedef ExpValuesTableQ25_103 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_25_103

    #if TINYMIND_USE_EXP_26_102
    template<>
    struct ExpTableValueSize<26, 102, true>
    {
        typedef ExpValuesTableQ26_102 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_26_102

    #if TINYMIND_USE_EXP_27_101
    template<>
    struct ExpTableValueSize<27, 101, true>
    {
        typedef ExpValuesTableQ27_101 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_27_101

    #if TINYMIND_USE_EXP_28_100
    template<>
    struct ExpTableValueSize<28, 100, true>
    {
        typedef ExpValuesTableQ28_100 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_28_100

    #if TINYMIND_USE_EXP_29_99
    template<>
    struct ExpTableValueSize<29, 99, true>
    {
        typedef ExpValuesTableQ29_99 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_29_99

    #if TINYMIND_USE_EXP_30_98
    template<>
    struct ExpTableValueSize<30, 98, true>
    {
        typedef ExpValuesTableQ30_98 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_30_98

    #if TINYMIND_USE_EXP_31_97
    template<>
    struct ExpTableValueSize<31, 97, true>
    {
        typedef ExpValuesTableQ31_97 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_31_97

    #if TINYMIND_USE_EXP_32_96
    template<>
    struct ExpTableValueSize<32, 96, true>
    {
        typedef ExpValuesTableQ32_96 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_32_96

    #if TINYMIND_USE_EXP_33_95
    template<>
    struct ExpTableValueSize<33, 95, true>
    {
        typedef ExpValuesTableQ33_95 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_33_95

    #if TINYMIND_USE_EXP_34_94
    template<>
    struct ExpTableValueSize<34, 94, true>
    {
        typedef ExpValuesTableQ34_94 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_34_94

    #if TINYMIND_USE_EXP_35_93
    template<>
    struct ExpTableValueSize<35, 93, true>
    {
        typedef ExpValuesTableQ35_93 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_35_93

    #if TINYMIND_USE_EXP_36_92
    template<>
    struct ExpTableValueSize<36, 92, true>
    {
        typedef ExpValuesTableQ36_92 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_36_92

    #if TINYMIND_USE_EXP_37_91
    template<>
    struct ExpTableValueSize<37, 91, true>
    {
        typedef ExpValuesTableQ37_91 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_37_91

    #if TINYMIND_USE_EXP_38_90
    template<>
    struct ExpTableValueSize<38, 90, true>
    {
        typedef ExpValuesTableQ38_90 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_38_90

    #if TINYMIND_USE_EXP_39_89
    template<>
    struct ExpTableValueSize<39, 89, true>
    {
        typedef ExpValuesTableQ39_89 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_39_89

    #if TINYMIND_USE_EXP_40_88
    template<>
    struct ExpTableValueSize<40, 88, true>
    {
        typedef ExpValuesTableQ40_88 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_40_88

    #if TINYMIND_USE_EXP_41_87
    template<>
    struct ExpTableValueSize<41, 87, true>
    {
        typedef ExpValuesTableQ41_87 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_41_87

    #if TINYMIND_USE_EXP_42_86
    template<>
    struct ExpTableValueSize<42, 86, true>
    {
        typedef ExpValuesTableQ42_86 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_42_86

    #if TINYMIND_USE_EXP_43_85
    template<>
    struct ExpTableValueSize<43, 85, true>
    {
        typedef ExpValuesTableQ43_85 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_43_85

    #if TINYMIND_USE_EXP_44_84
    template<>
    struct ExpTableValueSize<44, 84, true>
    {
        typedef ExpValuesTableQ44_84 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_44_84

    #if TINYMIND_USE_EXP_45_83
    template<>
    struct ExpTableValueSize<45, 83, true>
    {
        typedef ExpValuesTableQ45_83 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_45_83

    #if TINYMIND_USE_EXP_46_82
    template<>
    struct ExpTableValueSize<46, 82, true>
    {
        typedef ExpValuesTableQ46_82 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_46_82

    #if TINYMIND_USE_EXP_47_81
    template<>
    struct ExpTableValueSize<47, 81, true>
    {
        typedef ExpValuesTableQ47_81 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_47_81

    #if TINYMIND_USE_EXP_48_80
    template<>
    struct ExpTableValueSize<48, 80, true>
    {
        typedef ExpValuesTableQ48_80 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_48_80

    #if TINYMIND_USE_EXP_49_79
    template<>
    struct ExpTableValueSize<49, 79, true>
    {
        typedef ExpValuesTableQ49_79 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_49_79

    #if TINYMIND_USE_EXP_50_78
    template<>
    struct ExpTableValueSize<50, 78, true>
    {
        typedef ExpValuesTableQ50_78 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_50_78

    #if TINYMIND_USE_EXP_51_77
    template<>
    struct ExpTableValueSize<51, 77, true>
    {
        typedef ExpValuesTableQ51_77 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_51_77

    #if TINYMIND_USE_EXP_52_76
    template<>
    struct ExpTableValueSize<52, 76, true>
    {
        typedef ExpValuesTableQ52_76 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_52_76

    #if TINYMIND_USE_EXP_53_75
    template<>
    struct ExpTableValueSize<53, 75, true>
    {
        typedef ExpValuesTableQ53_75 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_53_75

    #if TINYMIND_USE_EXP_54_74
    template<>
    struct ExpTableValueSize<54, 74, true>
    {
        typedef ExpValuesTableQ54_74 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_54_74

    #if TINYMIND_USE_EXP_55_73
    template<>
    struct ExpTableValueSize<55, 73, true>
    {
        typedef ExpValuesTableQ55_73 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_55_73

    #if TINYMIND_USE_EXP_56_72
    template<>
    struct ExpTableValueSize<56, 72, true>
    {
        typedef ExpValuesTableQ56_72 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_56_72

    #if TINYMIND_USE_EXP_57_71
    template<>
    struct ExpTableValueSize<57, 71, true>
    {
        typedef ExpValuesTableQ57_71 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_57_71

    #if TINYMIND_USE_EXP_58_70
    template<>
    struct ExpTableValueSize<58, 70, true>
    {
        typedef ExpValuesTableQ58_70 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_58_70

    #if TINYMIND_USE_EXP_59_69
    template<>
    struct ExpTableValueSize<59, 69, true>
    {
        typedef ExpValuesTableQ59_69 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_59_69

    #if TINYMIND_USE_EXP_60_68
    template<>
    struct ExpTableValueSize<60, 68, true>
    {
        typedef ExpValuesTableQ60_68 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_60_68

    #if TINYMIND_USE_EXP_61_67
    template<>
    struct ExpTableValueSize<61, 67, true>
    {
        typedef ExpValuesTableQ61_67 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_61_67

    #if TINYMIND_USE_EXP_62_66
    template<>
    struct ExpTableValueSize<62, 66, true>
    {
        typedef ExpValuesTableQ62_66 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_62_66

    #if TINYMIND_USE_EXP_63_65
    template<>
    struct ExpTableValueSize<63, 65, true>
    {
        typedef ExpValuesTableQ63_65 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_63_65

    #if TINYMIND_USE_EXP_64_64
    template<>
    struct ExpTableValueSize<64, 64, true>
    {
        typedef ExpValuesTableQ64_64 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_64_64

    #if TINYMIND_USE_EXP_65_63
    template<>
    struct ExpTableValueSize<65, 63, true>
    {
        typedef ExpValuesTableQ65_63 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_65_63

    #if TINYMIND_USE_EXP_66_62
    template<>
    struct ExpTableValueSize<66, 62, true>
    {
        typedef ExpValuesTableQ66_62 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_66_62

    #if TINYMIND_USE_EXP_67_61
    template<>
    struct ExpTableValueSize<67, 61, true>
    {
        typedef ExpValuesTableQ67_61 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_67_61

    #if TINYMIND_USE_EXP_68_60
    template<>
    struct ExpTableValueSize<68, 60, true>
    {
        typedef ExpValuesTableQ68_60 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_68_60

    #if TINYMIND_USE_EXP_69_59
    template<>
    struct ExpTableValueSize<69, 59, true>
    {
        typedef ExpValuesTableQ69_59 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_69_59

    #if TINYMIND_USE_EXP_70_58
    template<>
    struct ExpTableValueSize<70, 58, true>
    {
        typedef ExpValuesTableQ70_58 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_70_58

    #if TINYMIND_USE_EXP_71_57
    template<>
    struct ExpTableValueSize<71, 57, true>
    {
        typedef ExpValuesTableQ71_57 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_71_57

    #if TINYMIND_USE_EXP_72_56
    template<>
    struct ExpTableValueSize<72, 56, true>
    {
        typedef ExpValuesTableQ72_56 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_72_56

    #if TINYMIND_USE_EXP_73_55
    template<>
    struct ExpTableValueSize<73, 55, true>
    {
        typedef ExpValuesTableQ73_55 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_73_55

    #if TINYMIND_USE_EXP_74_54
    template<>
    struct ExpTableValueSize<74, 54, true>
    {
        typedef ExpValuesTableQ74_54 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_74_54

    #if TINYMIND_USE_EXP_75_53
    template<>
    struct ExpTableValueSize<75, 53, true>
    {
        typedef ExpValuesTableQ75_53 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_75_53

    #if TINYMIND_USE_EXP_76_52
    template<>
    struct ExpTableValueSize<76, 52, true>
    {
        typedef ExpValuesTableQ76_52 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_76_52

    #if TINYMIND_USE_EXP_77_51
    template<>
    struct ExpTableValueSize<77, 51, true>
    {
        typedef ExpValuesTableQ77_51 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_77_51

    #if TINYMIND_USE_EXP_78_50
    template<>
    struct ExpTableValueSize<78, 50, true>
    {
        typedef ExpValuesTableQ78_50 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_78_50

    #if TINYMIND_USE_EXP_79_49
    template<>
    struct ExpTableValueSize<79, 49, true>
    {
        typedef ExpValuesTableQ79_49 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_79_49

    #if TINYMIND_USE_EXP_80_48
    template<>
    struct ExpTableValueSize<80, 48, true>
    {
        typedef ExpValuesTableQ80_48 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_80_48

    #if TINYMIND_USE_EXP_81_47
    template<>
    struct ExpTableValueSize<81, 47, true>
    {
        typedef ExpValuesTableQ81_47 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_81_47

    #if TINYMIND_USE_EXP_82_46
    template<>
    struct ExpTableValueSize<82, 46, true>
    {
        typedef ExpValuesTableQ82_46 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_82_46

    #if TINYMIND_USE_EXP_83_45
    template<>
    struct ExpTableValueSize<83, 45, true>
    {
        typedef ExpValuesTableQ83_45 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_83_45

    #if TINYMIND_USE_EXP_84_44
    template<>
    struct ExpTableValueSize<84, 44, true>
    {
        typedef ExpValuesTableQ84_44 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_84_44

    #if TINYMIND_USE_EXP_85_43
    template<>
    struct ExpTableValueSize<85, 43, true>
    {
        typedef ExpValuesTableQ85_43 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_85_43

    #if TINYMIND_USE_EXP_86_42
    template<>
    struct ExpTableValueSize<86, 42, true>
    {
        typedef ExpValuesTableQ86_42 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_86_42

    #if TINYMIND_USE_EXP_87_41
    template<>
    struct ExpTableValueSize<87, 41, true>
    {
        typedef ExpValuesTableQ87_41 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_87_41

    #if TINYMIND_USE_EXP_88_40
    template<>
    struct ExpTableValueSize<88, 40, true>
    {
        typedef ExpValuesTableQ88_40 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_88_40

    #if TINYMIND_USE_EXP_89_39
    template<>
    struct ExpTableValueSize<89, 39, true>
    {
        typedef ExpValuesTableQ89_39 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_89_39

    #if TINYMIND_USE_EXP_90_38
    template<>
    struct ExpTableValueSize<90, 38, true>
    {
        typedef ExpValuesTableQ90_38 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_90_38

    #if TINYMIND_USE_EXP_91_37
    template<>
    struct ExpTableValueSize<91, 37, true>
    {
        typedef ExpValuesTableQ91_37 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_91_37

    #if TINYMIND_USE_EXP_92_36
    template<>
    struct ExpTableValueSize<92, 36, true>
    {
        typedef ExpValuesTableQ92_36 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_92_36

    #if TINYMIND_USE_EXP_93_35
    template<>
    struct ExpTableValueSize<93, 35, true>
    {
        typedef ExpValuesTableQ93_35 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_93_35

    #if TINYMIND_USE_EXP_94_34
    template<>
    struct ExpTableValueSize<94, 34, true>
    {
        typedef ExpValuesTableQ94_34 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_94_34

    #if TINYMIND_USE_EXP_95_33
    template<>
    struct ExpTableValueSize<95, 33, true>
    {
        typedef ExpValuesTableQ95_33 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_95_33

    #if TINYMIND_USE_EXP_96_32
    template<>
    struct ExpTableValueSize<96, 32, true>
    {
        typedef ExpValuesTableQ96_32 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_96_32

    #if TINYMIND_USE_EXP_97_31
    template<>
    struct ExpTableValueSize<97, 31, true>
    {
        typedef ExpValuesTableQ97_31 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_97_31

    #if TINYMIND_USE_EXP_98_30
    template<>
    struct ExpTableValueSize<98, 30, true>
    {
        typedef ExpValuesTableQ98_30 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_98_30

    #if TINYMIND_USE_EXP_99_29
    template<>
    struct ExpTableValueSize<99, 29, true>
    {
        typedef ExpValuesTableQ99_29 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_99_29

    #if TINYMIND_USE_EXP_100_28
    template<>
    struct ExpTableValueSize<100, 28, true>
    {
        typedef ExpValuesTableQ100_28 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_100_28

    #if TINYMIND_USE_EXP_101_27
    template<>
    struct ExpTableValueSize<101, 27, true>
    {
        typedef ExpValuesTableQ101_27 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_101_27

    #if TINYMIND_USE_EXP_102_26
    template<>
    struct ExpTableValueSize<102, 26, true>
    {
        typedef ExpValuesTableQ102_26 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_102_26

    #if TINYMIND_USE_EXP_103_25
    template<>
    struct ExpTableValueSize<103, 25, true>
    {
        typedef ExpValuesTableQ103_25 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_103_25

    #if TINYMIND_USE_EXP_104_24
    template<>
    struct ExpTableValueSize<104, 24, true>
    {
        typedef ExpValuesTableQ104_24 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_104_24

    #if TINYMIND_USE_EXP_105_23
    template<>
    struct ExpTableValueSize<105, 23, true>
    {
        typedef ExpValuesTableQ105_23 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_105_23

    #if TINYMIND_USE_EXP_106_22
    template<>
    struct ExpTableValueSize<106, 22, true>
    {
        typedef ExpValuesTableQ106_22 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_106_22

    #if TINYMIND_USE_EXP_107_21
    template<>
    struct ExpTableValueSize<107, 21, true>
    {
        typedef ExpValuesTableQ107_21 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_107_21

    #if TINYMIND_USE_EXP_108_20
    template<>
    struct ExpTableValueSize<108, 20, true>
    {
        typedef ExpValuesTableQ108_20 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_108_20

    #if TINYMIND_USE_EXP_109_19
    template<>
    struct ExpTableValueSize<109, 19, true>
    {
        typedef ExpValuesTableQ109_19 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_109_19

    #if TINYMIND_USE_EXP_110_18
    template<>
    struct ExpTableValueSize<110, 18, true>
    {
        typedef ExpValuesTableQ110_18 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_110_18

    #if TINYMIND_USE_EXP_111_17
    template<>
    struct ExpTableValueSize<111, 17, true>
    {
        typedef ExpValuesTableQ111_17 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_111_17

    #if TINYMIND_USE_EXP_112_16
    template<>
    struct ExpTableValueSize<112, 16, true>
    {
        typedef ExpValuesTableQ112_16 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_112_16

    #if TINYMIND_USE_EXP_113_15
    template<>
    struct ExpTableValueSize<113, 15, true>
    {
        typedef ExpValuesTableQ113_15 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_113_15

    #if TINYMIND_USE_EXP_114_14
    template<>
    struct ExpTableValueSize<114, 14, true>
    {
        typedef ExpValuesTableQ114_14 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_114_14

    #if TINYMIND_USE_EXP_115_13
    template<>
    struct ExpTableValueSize<115, 13, true>
    {
        typedef ExpValuesTableQ115_13 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_115_13

    #if TINYMIND_USE_EXP_116_12
    template<>
    struct ExpTableValueSize<116, 12, true>
    {
        typedef ExpValuesTableQ116_12 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_116_12

    #if TINYMIND_USE_EXP_117_11
    template<>
    struct ExpTableValueSize<117, 11, true>
    {
        typedef ExpValuesTableQ117_11 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_117_11

    #if TINYMIND_USE_EXP_118_10
    template<>
    struct ExpTableValueSize<118, 10, true>
    {
        typedef ExpValuesTableQ118_10 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_118_10

    #if TINYMIND_USE_EXP_119_9
    template<>
    struct ExpTableValueSize<119, 9, true>
    {
        typedef ExpValuesTableQ119_9 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_119_9

    #if TINYMIND_USE_EXP_120_8
    template<>
    struct ExpTableValueSize<120, 8, true>
    {
        typedef ExpValuesTableQ120_8 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_120_8

    #if TINYMIND_USE_EXP_121_7
    template<>
    struct ExpTableValueSize<121, 7, true>
    {
        typedef ExpValuesTableQ121_7 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_121_7

    #if TINYMIND_USE_EXP_122_6
    template<>
    struct ExpTableValueSize<122, 6, true>
    {
        typedef ExpValuesTableQ122_6 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_122_6

    #if TINYMIND_USE_EXP_123_5
    template<>
    struct ExpTableValueSize<123, 5, true>
    {
        typedef ExpValuesTableQ123_5 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_123_5

    #if TINYMIND_USE_EXP_124_4
    template<>
    struct ExpTableValueSize<124, 4, true>
    {
        typedef ExpValuesTableQ124_4 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_124_4

    #if TINYMIND_USE_EXP_125_3
    template<>
    struct ExpTableValueSize<125, 3, true>
    {
        typedef ExpValuesTableQ125_3 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_125_3

    #if TINYMIND_USE_EXP_126_2
    template<>
    struct ExpTableValueSize<126, 2, true>
    {
        typedef ExpValuesTableQ126_2 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_126_2

    #if TINYMIND_USE_EXP_127_1
    template<>
    struct ExpTableValueSize<127, 1, true>
    {
        typedef ExpValuesTableQ127_1 ExpTableType;
    };
    #endif // TINYMIND_USE_EXP_127_1

    template<unsigned FixedBits,unsigned FracBits, bool IsSigned>
    struct ExpValuesTableSelector
    {
        typedef typename ExpTableValueSize<FixedBits, FracBits, IsSigned>::ExpTableType ExpTableType;
    };
}
