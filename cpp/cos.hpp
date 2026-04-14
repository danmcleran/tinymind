/**
* Copyright (c) 2026 Dan McLeran
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

#include "cosValues8Bit.hpp"
#include "cosValues16Bit.hpp"
#include "cosValues32Bit.hpp"
#include "cosValues64Bit.hpp"
#include "cosValues128Bit.hpp"

namespace tinymind {
    template<unsigned FixedBits, unsigned FracBits, bool IsSigned>
    struct CosTableValueSize
    {
    };

    #if TINYMIND_USE_COS_1_7
    template<>
    struct CosTableValueSize<1, 7, true>
    {
        typedef CosValuesTableQ1_7 CosTableType;
    };
    #endif // TINYMIND_USE_COS_1_7

    #if TINYMIND_USE_COS_2_6
    template<>
    struct CosTableValueSize<2, 6, true>
    {
        typedef CosValuesTableQ2_6 CosTableType;
    };
    #endif // TINYMIND_USE_COS_2_6

    #if TINYMIND_USE_COS_3_5
    template<>
    struct CosTableValueSize<3, 5, true>
    {
        typedef CosValuesTableQ3_5 CosTableType;
    };
    #endif // TINYMIND_USE_COS_3_5

    #if TINYMIND_USE_COS_4_4
    template<>
    struct CosTableValueSize<4, 4, true>
    {
        typedef CosValuesTableQ4_4 CosTableType;
    };
    #endif // TINYMIND_USE_COS_4_4

    #if TINYMIND_USE_COS_5_3
    template<>
    struct CosTableValueSize<5, 3, true>
    {
        typedef CosValuesTableQ5_3 CosTableType;
    };
    #endif // TINYMIND_USE_COS_5_3

    #if TINYMIND_USE_COS_6_2
    template<>
    struct CosTableValueSize<6, 2, true>
    {
        typedef CosValuesTableQ6_2 CosTableType;
    };
    #endif // TINYMIND_USE_COS_6_2

    #if TINYMIND_USE_COS_7_1
    template<>
    struct CosTableValueSize<7, 1, true>
    {
        typedef CosValuesTableQ7_1 CosTableType;
    };
    #endif // TINYMIND_USE_COS_7_1

    #if TINYMIND_USE_COS_1_15
    template<>
    struct CosTableValueSize<1, 15, true>
    {
        typedef CosValuesTableQ1_15 CosTableType;
    };
    #endif // TINYMIND_USE_COS_1_15

    #if TINYMIND_USE_COS_2_14
    template<>
    struct CosTableValueSize<2, 14, true>
    {
        typedef CosValuesTableQ2_14 CosTableType;
    };
    #endif // TINYMIND_USE_COS_2_14

    #if TINYMIND_USE_COS_3_13
    template<>
    struct CosTableValueSize<3, 13, true>
    {
        typedef CosValuesTableQ3_13 CosTableType;
    };
    #endif // TINYMIND_USE_COS_3_13

    #if TINYMIND_USE_COS_4_12
    template<>
    struct CosTableValueSize<4, 12, true>
    {
        typedef CosValuesTableQ4_12 CosTableType;
    };
    #endif // TINYMIND_USE_COS_4_12

    #if TINYMIND_USE_COS_5_11
    template<>
    struct CosTableValueSize<5, 11, true>
    {
        typedef CosValuesTableQ5_11 CosTableType;
    };
    #endif // TINYMIND_USE_COS_5_11

    #if TINYMIND_USE_COS_6_10
    template<>
    struct CosTableValueSize<6, 10, true>
    {
        typedef CosValuesTableQ6_10 CosTableType;
    };
    #endif // TINYMIND_USE_COS_6_10

    #if TINYMIND_USE_COS_7_9
    template<>
    struct CosTableValueSize<7, 9, true>
    {
        typedef CosValuesTableQ7_9 CosTableType;
    };
    #endif // TINYMIND_USE_COS_7_9

    #if TINYMIND_USE_COS_8_8
    template<>
    struct CosTableValueSize<8, 8, true>
    {
        typedef CosValuesTableQ8_8 CosTableType;
    };
    #endif // TINYMIND_USE_COS_8_8

    #if TINYMIND_USE_COS_9_7
    template<>
    struct CosTableValueSize<9, 7, true>
    {
        typedef CosValuesTableQ9_7 CosTableType;
    };
    #endif // TINYMIND_USE_COS_9_7

    #if TINYMIND_USE_COS_10_6
    template<>
    struct CosTableValueSize<10, 6, true>
    {
        typedef CosValuesTableQ10_6 CosTableType;
    };
    #endif // TINYMIND_USE_COS_10_6

    #if TINYMIND_USE_COS_11_5
    template<>
    struct CosTableValueSize<11, 5, true>
    {
        typedef CosValuesTableQ11_5 CosTableType;
    };
    #endif // TINYMIND_USE_COS_11_5

    #if TINYMIND_USE_COS_12_4
    template<>
    struct CosTableValueSize<12, 4, true>
    {
        typedef CosValuesTableQ12_4 CosTableType;
    };
    #endif // TINYMIND_USE_COS_12_4

    #if TINYMIND_USE_COS_13_3
    template<>
    struct CosTableValueSize<13, 3, true>
    {
        typedef CosValuesTableQ13_3 CosTableType;
    };
    #endif // TINYMIND_USE_COS_13_3

    #if TINYMIND_USE_COS_14_2
    template<>
    struct CosTableValueSize<14, 2, true>
    {
        typedef CosValuesTableQ14_2 CosTableType;
    };
    #endif // TINYMIND_USE_COS_14_2

    #if TINYMIND_USE_COS_15_1
    template<>
    struct CosTableValueSize<15, 1, true>
    {
        typedef CosValuesTableQ15_1 CosTableType;
    };
    #endif // TINYMIND_USE_COS_15_1

    #if TINYMIND_USE_COS_1_31
    template<>
    struct CosTableValueSize<1, 31, true>
    {
        typedef CosValuesTableQ1_31 CosTableType;
    };
    #endif // TINYMIND_USE_COS_1_31

    #if TINYMIND_USE_COS_2_30
    template<>
    struct CosTableValueSize<2, 30, true>
    {
        typedef CosValuesTableQ2_30 CosTableType;
    };
    #endif // TINYMIND_USE_COS_2_30

    #if TINYMIND_USE_COS_3_29
    template<>
    struct CosTableValueSize<3, 29, true>
    {
        typedef CosValuesTableQ3_29 CosTableType;
    };
    #endif // TINYMIND_USE_COS_3_29

    #if TINYMIND_USE_COS_4_28
    template<>
    struct CosTableValueSize<4, 28, true>
    {
        typedef CosValuesTableQ4_28 CosTableType;
    };
    #endif // TINYMIND_USE_COS_4_28

    #if TINYMIND_USE_COS_5_27
    template<>
    struct CosTableValueSize<5, 27, true>
    {
        typedef CosValuesTableQ5_27 CosTableType;
    };
    #endif // TINYMIND_USE_COS_5_27

    #if TINYMIND_USE_COS_6_26
    template<>
    struct CosTableValueSize<6, 26, true>
    {
        typedef CosValuesTableQ6_26 CosTableType;
    };
    #endif // TINYMIND_USE_COS_6_26

    #if TINYMIND_USE_COS_7_25
    template<>
    struct CosTableValueSize<7, 25, true>
    {
        typedef CosValuesTableQ7_25 CosTableType;
    };
    #endif // TINYMIND_USE_COS_7_25

    #if TINYMIND_USE_COS_8_24
    template<>
    struct CosTableValueSize<8, 24, true>
    {
        typedef CosValuesTableQ8_24 CosTableType;
    };
    #endif // TINYMIND_USE_COS_8_24

    #if TINYMIND_USE_COS_9_23
    template<>
    struct CosTableValueSize<9, 23, true>
    {
        typedef CosValuesTableQ9_23 CosTableType;
    };
    #endif // TINYMIND_USE_COS_9_23

    #if TINYMIND_USE_COS_10_22
    template<>
    struct CosTableValueSize<10, 22, true>
    {
        typedef CosValuesTableQ10_22 CosTableType;
    };
    #endif // TINYMIND_USE_COS_10_22

    #if TINYMIND_USE_COS_11_21
    template<>
    struct CosTableValueSize<11, 21, true>
    {
        typedef CosValuesTableQ11_21 CosTableType;
    };
    #endif // TINYMIND_USE_COS_11_21

    #if TINYMIND_USE_COS_12_20
    template<>
    struct CosTableValueSize<12, 20, true>
    {
        typedef CosValuesTableQ12_20 CosTableType;
    };
    #endif // TINYMIND_USE_COS_12_20

    #if TINYMIND_USE_COS_13_19
    template<>
    struct CosTableValueSize<13, 19, true>
    {
        typedef CosValuesTableQ13_19 CosTableType;
    };
    #endif // TINYMIND_USE_COS_13_19

    #if TINYMIND_USE_COS_14_18
    template<>
    struct CosTableValueSize<14, 18, true>
    {
        typedef CosValuesTableQ14_18 CosTableType;
    };
    #endif // TINYMIND_USE_COS_14_18

    #if TINYMIND_USE_COS_15_17
    template<>
    struct CosTableValueSize<15, 17, true>
    {
        typedef CosValuesTableQ15_17 CosTableType;
    };
    #endif // TINYMIND_USE_COS_15_17

    #if TINYMIND_USE_COS_16_16
    template<>
    struct CosTableValueSize<16, 16, true>
    {
        typedef CosValuesTableQ16_16 CosTableType;
    };
    #endif // TINYMIND_USE_COS_16_16

    #if TINYMIND_USE_COS_17_15
    template<>
    struct CosTableValueSize<17, 15, true>
    {
        typedef CosValuesTableQ17_15 CosTableType;
    };
    #endif // TINYMIND_USE_COS_17_15

    #if TINYMIND_USE_COS_18_14
    template<>
    struct CosTableValueSize<18, 14, true>
    {
        typedef CosValuesTableQ18_14 CosTableType;
    };
    #endif // TINYMIND_USE_COS_18_14

    #if TINYMIND_USE_COS_19_13
    template<>
    struct CosTableValueSize<19, 13, true>
    {
        typedef CosValuesTableQ19_13 CosTableType;
    };
    #endif // TINYMIND_USE_COS_19_13

    #if TINYMIND_USE_COS_20_12
    template<>
    struct CosTableValueSize<20, 12, true>
    {
        typedef CosValuesTableQ20_12 CosTableType;
    };
    #endif // TINYMIND_USE_COS_20_12

    #if TINYMIND_USE_COS_21_11
    template<>
    struct CosTableValueSize<21, 11, true>
    {
        typedef CosValuesTableQ21_11 CosTableType;
    };
    #endif // TINYMIND_USE_COS_21_11

    #if TINYMIND_USE_COS_22_10
    template<>
    struct CosTableValueSize<22, 10, true>
    {
        typedef CosValuesTableQ22_10 CosTableType;
    };
    #endif // TINYMIND_USE_COS_22_10

    #if TINYMIND_USE_COS_23_9
    template<>
    struct CosTableValueSize<23, 9, true>
    {
        typedef CosValuesTableQ23_9 CosTableType;
    };
    #endif // TINYMIND_USE_COS_23_9

    #if TINYMIND_USE_COS_24_8
    template<>
    struct CosTableValueSize<24, 8, true>
    {
        typedef CosValuesTableQ24_8 CosTableType;
    };
    #endif // TINYMIND_USE_COS_24_8

    #if TINYMIND_USE_COS_25_7
    template<>
    struct CosTableValueSize<25, 7, true>
    {
        typedef CosValuesTableQ25_7 CosTableType;
    };
    #endif // TINYMIND_USE_COS_25_7

    #if TINYMIND_USE_COS_26_6
    template<>
    struct CosTableValueSize<26, 6, true>
    {
        typedef CosValuesTableQ26_6 CosTableType;
    };
    #endif // TINYMIND_USE_COS_26_6

    #if TINYMIND_USE_COS_27_5
    template<>
    struct CosTableValueSize<27, 5, true>
    {
        typedef CosValuesTableQ27_5 CosTableType;
    };
    #endif // TINYMIND_USE_COS_27_5

    #if TINYMIND_USE_COS_28_4
    template<>
    struct CosTableValueSize<28, 4, true>
    {
        typedef CosValuesTableQ28_4 CosTableType;
    };
    #endif // TINYMIND_USE_COS_28_4

    #if TINYMIND_USE_COS_29_3
    template<>
    struct CosTableValueSize<29, 3, true>
    {
        typedef CosValuesTableQ29_3 CosTableType;
    };
    #endif // TINYMIND_USE_COS_29_3

    #if TINYMIND_USE_COS_30_2
    template<>
    struct CosTableValueSize<30, 2, true>
    {
        typedef CosValuesTableQ30_2 CosTableType;
    };
    #endif // TINYMIND_USE_COS_30_2

    #if TINYMIND_USE_COS_31_1
    template<>
    struct CosTableValueSize<31, 1, true>
    {
        typedef CosValuesTableQ31_1 CosTableType;
    };
    #endif // TINYMIND_USE_COS_31_1

    #if TINYMIND_USE_COS_1_63
    template<>
    struct CosTableValueSize<1, 63, true>
    {
        typedef CosValuesTableQ1_63 CosTableType;
    };
    #endif // TINYMIND_USE_COS_1_63

    #if TINYMIND_USE_COS_2_62
    template<>
    struct CosTableValueSize<2, 62, true>
    {
        typedef CosValuesTableQ2_62 CosTableType;
    };
    #endif // TINYMIND_USE_COS_2_62

    #if TINYMIND_USE_COS_3_61
    template<>
    struct CosTableValueSize<3, 61, true>
    {
        typedef CosValuesTableQ3_61 CosTableType;
    };
    #endif // TINYMIND_USE_COS_3_61

    #if TINYMIND_USE_COS_4_60
    template<>
    struct CosTableValueSize<4, 60, true>
    {
        typedef CosValuesTableQ4_60 CosTableType;
    };
    #endif // TINYMIND_USE_COS_4_60

    #if TINYMIND_USE_COS_5_59
    template<>
    struct CosTableValueSize<5, 59, true>
    {
        typedef CosValuesTableQ5_59 CosTableType;
    };
    #endif // TINYMIND_USE_COS_5_59

    #if TINYMIND_USE_COS_6_58
    template<>
    struct CosTableValueSize<6, 58, true>
    {
        typedef CosValuesTableQ6_58 CosTableType;
    };
    #endif // TINYMIND_USE_COS_6_58

    #if TINYMIND_USE_COS_7_57
    template<>
    struct CosTableValueSize<7, 57, true>
    {
        typedef CosValuesTableQ7_57 CosTableType;
    };
    #endif // TINYMIND_USE_COS_7_57

    #if TINYMIND_USE_COS_8_56
    template<>
    struct CosTableValueSize<8, 56, true>
    {
        typedef CosValuesTableQ8_56 CosTableType;
    };
    #endif // TINYMIND_USE_COS_8_56

    #if TINYMIND_USE_COS_9_55
    template<>
    struct CosTableValueSize<9, 55, true>
    {
        typedef CosValuesTableQ9_55 CosTableType;
    };
    #endif // TINYMIND_USE_COS_9_55

    #if TINYMIND_USE_COS_10_54
    template<>
    struct CosTableValueSize<10, 54, true>
    {
        typedef CosValuesTableQ10_54 CosTableType;
    };
    #endif // TINYMIND_USE_COS_10_54

    #if TINYMIND_USE_COS_11_53
    template<>
    struct CosTableValueSize<11, 53, true>
    {
        typedef CosValuesTableQ11_53 CosTableType;
    };
    #endif // TINYMIND_USE_COS_11_53

    #if TINYMIND_USE_COS_12_52
    template<>
    struct CosTableValueSize<12, 52, true>
    {
        typedef CosValuesTableQ12_52 CosTableType;
    };
    #endif // TINYMIND_USE_COS_12_52

    #if TINYMIND_USE_COS_13_51
    template<>
    struct CosTableValueSize<13, 51, true>
    {
        typedef CosValuesTableQ13_51 CosTableType;
    };
    #endif // TINYMIND_USE_COS_13_51

    #if TINYMIND_USE_COS_14_50
    template<>
    struct CosTableValueSize<14, 50, true>
    {
        typedef CosValuesTableQ14_50 CosTableType;
    };
    #endif // TINYMIND_USE_COS_14_50

    #if TINYMIND_USE_COS_15_49
    template<>
    struct CosTableValueSize<15, 49, true>
    {
        typedef CosValuesTableQ15_49 CosTableType;
    };
    #endif // TINYMIND_USE_COS_15_49

    #if TINYMIND_USE_COS_16_48
    template<>
    struct CosTableValueSize<16, 48, true>
    {
        typedef CosValuesTableQ16_48 CosTableType;
    };
    #endif // TINYMIND_USE_COS_16_48

    #if TINYMIND_USE_COS_17_47
    template<>
    struct CosTableValueSize<17, 47, true>
    {
        typedef CosValuesTableQ17_47 CosTableType;
    };
    #endif // TINYMIND_USE_COS_17_47

    #if TINYMIND_USE_COS_18_46
    template<>
    struct CosTableValueSize<18, 46, true>
    {
        typedef CosValuesTableQ18_46 CosTableType;
    };
    #endif // TINYMIND_USE_COS_18_46

    #if TINYMIND_USE_COS_19_45
    template<>
    struct CosTableValueSize<19, 45, true>
    {
        typedef CosValuesTableQ19_45 CosTableType;
    };
    #endif // TINYMIND_USE_COS_19_45

    #if TINYMIND_USE_COS_20_44
    template<>
    struct CosTableValueSize<20, 44, true>
    {
        typedef CosValuesTableQ20_44 CosTableType;
    };
    #endif // TINYMIND_USE_COS_20_44

    #if TINYMIND_USE_COS_21_43
    template<>
    struct CosTableValueSize<21, 43, true>
    {
        typedef CosValuesTableQ21_43 CosTableType;
    };
    #endif // TINYMIND_USE_COS_21_43

    #if TINYMIND_USE_COS_22_42
    template<>
    struct CosTableValueSize<22, 42, true>
    {
        typedef CosValuesTableQ22_42 CosTableType;
    };
    #endif // TINYMIND_USE_COS_22_42

    #if TINYMIND_USE_COS_23_41
    template<>
    struct CosTableValueSize<23, 41, true>
    {
        typedef CosValuesTableQ23_41 CosTableType;
    };
    #endif // TINYMIND_USE_COS_23_41

    #if TINYMIND_USE_COS_24_40
    template<>
    struct CosTableValueSize<24, 40, true>
    {
        typedef CosValuesTableQ24_40 CosTableType;
    };
    #endif // TINYMIND_USE_COS_24_40

    #if TINYMIND_USE_COS_25_39
    template<>
    struct CosTableValueSize<25, 39, true>
    {
        typedef CosValuesTableQ25_39 CosTableType;
    };
    #endif // TINYMIND_USE_COS_25_39

    #if TINYMIND_USE_COS_26_38
    template<>
    struct CosTableValueSize<26, 38, true>
    {
        typedef CosValuesTableQ26_38 CosTableType;
    };
    #endif // TINYMIND_USE_COS_26_38

    #if TINYMIND_USE_COS_27_37
    template<>
    struct CosTableValueSize<27, 37, true>
    {
        typedef CosValuesTableQ27_37 CosTableType;
    };
    #endif // TINYMIND_USE_COS_27_37

    #if TINYMIND_USE_COS_28_36
    template<>
    struct CosTableValueSize<28, 36, true>
    {
        typedef CosValuesTableQ28_36 CosTableType;
    };
    #endif // TINYMIND_USE_COS_28_36

    #if TINYMIND_USE_COS_29_35
    template<>
    struct CosTableValueSize<29, 35, true>
    {
        typedef CosValuesTableQ29_35 CosTableType;
    };
    #endif // TINYMIND_USE_COS_29_35

    #if TINYMIND_USE_COS_30_34
    template<>
    struct CosTableValueSize<30, 34, true>
    {
        typedef CosValuesTableQ30_34 CosTableType;
    };
    #endif // TINYMIND_USE_COS_30_34

    #if TINYMIND_USE_COS_31_33
    template<>
    struct CosTableValueSize<31, 33, true>
    {
        typedef CosValuesTableQ31_33 CosTableType;
    };
    #endif // TINYMIND_USE_COS_31_33

    #if TINYMIND_USE_COS_32_32
    template<>
    struct CosTableValueSize<32, 32, true>
    {
        typedef CosValuesTableQ32_32 CosTableType;
    };
    #endif // TINYMIND_USE_COS_32_32

    #if TINYMIND_USE_COS_33_31
    template<>
    struct CosTableValueSize<33, 31, true>
    {
        typedef CosValuesTableQ33_31 CosTableType;
    };
    #endif // TINYMIND_USE_COS_33_31

    #if TINYMIND_USE_COS_34_30
    template<>
    struct CosTableValueSize<34, 30, true>
    {
        typedef CosValuesTableQ34_30 CosTableType;
    };
    #endif // TINYMIND_USE_COS_34_30

    #if TINYMIND_USE_COS_35_29
    template<>
    struct CosTableValueSize<35, 29, true>
    {
        typedef CosValuesTableQ35_29 CosTableType;
    };
    #endif // TINYMIND_USE_COS_35_29

    #if TINYMIND_USE_COS_36_28
    template<>
    struct CosTableValueSize<36, 28, true>
    {
        typedef CosValuesTableQ36_28 CosTableType;
    };
    #endif // TINYMIND_USE_COS_36_28

    #if TINYMIND_USE_COS_37_27
    template<>
    struct CosTableValueSize<37, 27, true>
    {
        typedef CosValuesTableQ37_27 CosTableType;
    };
    #endif // TINYMIND_USE_COS_37_27

    #if TINYMIND_USE_COS_38_26
    template<>
    struct CosTableValueSize<38, 26, true>
    {
        typedef CosValuesTableQ38_26 CosTableType;
    };
    #endif // TINYMIND_USE_COS_38_26

    #if TINYMIND_USE_COS_39_25
    template<>
    struct CosTableValueSize<39, 25, true>
    {
        typedef CosValuesTableQ39_25 CosTableType;
    };
    #endif // TINYMIND_USE_COS_39_25

    #if TINYMIND_USE_COS_40_24
    template<>
    struct CosTableValueSize<40, 24, true>
    {
        typedef CosValuesTableQ40_24 CosTableType;
    };
    #endif // TINYMIND_USE_COS_40_24

    #if TINYMIND_USE_COS_41_23
    template<>
    struct CosTableValueSize<41, 23, true>
    {
        typedef CosValuesTableQ41_23 CosTableType;
    };
    #endif // TINYMIND_USE_COS_41_23

    #if TINYMIND_USE_COS_42_22
    template<>
    struct CosTableValueSize<42, 22, true>
    {
        typedef CosValuesTableQ42_22 CosTableType;
    };
    #endif // TINYMIND_USE_COS_42_22

    #if TINYMIND_USE_COS_43_21
    template<>
    struct CosTableValueSize<43, 21, true>
    {
        typedef CosValuesTableQ43_21 CosTableType;
    };
    #endif // TINYMIND_USE_COS_43_21

    #if TINYMIND_USE_COS_44_20
    template<>
    struct CosTableValueSize<44, 20, true>
    {
        typedef CosValuesTableQ44_20 CosTableType;
    };
    #endif // TINYMIND_USE_COS_44_20

    #if TINYMIND_USE_COS_45_19
    template<>
    struct CosTableValueSize<45, 19, true>
    {
        typedef CosValuesTableQ45_19 CosTableType;
    };
    #endif // TINYMIND_USE_COS_45_19

    #if TINYMIND_USE_COS_46_18
    template<>
    struct CosTableValueSize<46, 18, true>
    {
        typedef CosValuesTableQ46_18 CosTableType;
    };
    #endif // TINYMIND_USE_COS_46_18

    #if TINYMIND_USE_COS_47_17
    template<>
    struct CosTableValueSize<47, 17, true>
    {
        typedef CosValuesTableQ47_17 CosTableType;
    };
    #endif // TINYMIND_USE_COS_47_17

    #if TINYMIND_USE_COS_48_16
    template<>
    struct CosTableValueSize<48, 16, true>
    {
        typedef CosValuesTableQ48_16 CosTableType;
    };
    #endif // TINYMIND_USE_COS_48_16

    #if TINYMIND_USE_COS_49_15
    template<>
    struct CosTableValueSize<49, 15, true>
    {
        typedef CosValuesTableQ49_15 CosTableType;
    };
    #endif // TINYMIND_USE_COS_49_15

    #if TINYMIND_USE_COS_50_14
    template<>
    struct CosTableValueSize<50, 14, true>
    {
        typedef CosValuesTableQ50_14 CosTableType;
    };
    #endif // TINYMIND_USE_COS_50_14

    #if TINYMIND_USE_COS_51_13
    template<>
    struct CosTableValueSize<51, 13, true>
    {
        typedef CosValuesTableQ51_13 CosTableType;
    };
    #endif // TINYMIND_USE_COS_51_13

    #if TINYMIND_USE_COS_52_12
    template<>
    struct CosTableValueSize<52, 12, true>
    {
        typedef CosValuesTableQ52_12 CosTableType;
    };
    #endif // TINYMIND_USE_COS_52_12

    #if TINYMIND_USE_COS_53_11
    template<>
    struct CosTableValueSize<53, 11, true>
    {
        typedef CosValuesTableQ53_11 CosTableType;
    };
    #endif // TINYMIND_USE_COS_53_11

    #if TINYMIND_USE_COS_54_10
    template<>
    struct CosTableValueSize<54, 10, true>
    {
        typedef CosValuesTableQ54_10 CosTableType;
    };
    #endif // TINYMIND_USE_COS_54_10

    #if TINYMIND_USE_COS_55_9
    template<>
    struct CosTableValueSize<55, 9, true>
    {
        typedef CosValuesTableQ55_9 CosTableType;
    };
    #endif // TINYMIND_USE_COS_55_9

    #if TINYMIND_USE_COS_56_8
    template<>
    struct CosTableValueSize<56, 8, true>
    {
        typedef CosValuesTableQ56_8 CosTableType;
    };
    #endif // TINYMIND_USE_COS_56_8

    #if TINYMIND_USE_COS_57_7
    template<>
    struct CosTableValueSize<57, 7, true>
    {
        typedef CosValuesTableQ57_7 CosTableType;
    };
    #endif // TINYMIND_USE_COS_57_7

    #if TINYMIND_USE_COS_58_6
    template<>
    struct CosTableValueSize<58, 6, true>
    {
        typedef CosValuesTableQ58_6 CosTableType;
    };
    #endif // TINYMIND_USE_COS_58_6

    #if TINYMIND_USE_COS_59_5
    template<>
    struct CosTableValueSize<59, 5, true>
    {
        typedef CosValuesTableQ59_5 CosTableType;
    };
    #endif // TINYMIND_USE_COS_59_5

    #if TINYMIND_USE_COS_60_4
    template<>
    struct CosTableValueSize<60, 4, true>
    {
        typedef CosValuesTableQ60_4 CosTableType;
    };
    #endif // TINYMIND_USE_COS_60_4

    #if TINYMIND_USE_COS_61_3
    template<>
    struct CosTableValueSize<61, 3, true>
    {
        typedef CosValuesTableQ61_3 CosTableType;
    };
    #endif // TINYMIND_USE_COS_61_3

    #if TINYMIND_USE_COS_62_2
    template<>
    struct CosTableValueSize<62, 2, true>
    {
        typedef CosValuesTableQ62_2 CosTableType;
    };
    #endif // TINYMIND_USE_COS_62_2

    #if TINYMIND_USE_COS_63_1
    template<>
    struct CosTableValueSize<63, 1, true>
    {
        typedef CosValuesTableQ63_1 CosTableType;
    };
    #endif // TINYMIND_USE_COS_63_1

    #if TINYMIND_USE_COS_1_127
    template<>
    struct CosTableValueSize<1, 127, true>
    {
        typedef CosValuesTableQ1_127 CosTableType;
    };
    #endif // TINYMIND_USE_COS_1_127

    #if TINYMIND_USE_COS_2_126
    template<>
    struct CosTableValueSize<2, 126, true>
    {
        typedef CosValuesTableQ2_126 CosTableType;
    };
    #endif // TINYMIND_USE_COS_2_126

    #if TINYMIND_USE_COS_3_125
    template<>
    struct CosTableValueSize<3, 125, true>
    {
        typedef CosValuesTableQ3_125 CosTableType;
    };
    #endif // TINYMIND_USE_COS_3_125

    #if TINYMIND_USE_COS_4_124
    template<>
    struct CosTableValueSize<4, 124, true>
    {
        typedef CosValuesTableQ4_124 CosTableType;
    };
    #endif // TINYMIND_USE_COS_4_124

    #if TINYMIND_USE_COS_5_123
    template<>
    struct CosTableValueSize<5, 123, true>
    {
        typedef CosValuesTableQ5_123 CosTableType;
    };
    #endif // TINYMIND_USE_COS_5_123

    #if TINYMIND_USE_COS_6_122
    template<>
    struct CosTableValueSize<6, 122, true>
    {
        typedef CosValuesTableQ6_122 CosTableType;
    };
    #endif // TINYMIND_USE_COS_6_122

    #if TINYMIND_USE_COS_7_121
    template<>
    struct CosTableValueSize<7, 121, true>
    {
        typedef CosValuesTableQ7_121 CosTableType;
    };
    #endif // TINYMIND_USE_COS_7_121

    #if TINYMIND_USE_COS_8_120
    template<>
    struct CosTableValueSize<8, 120, true>
    {
        typedef CosValuesTableQ8_120 CosTableType;
    };
    #endif // TINYMIND_USE_COS_8_120

    #if TINYMIND_USE_COS_9_119
    template<>
    struct CosTableValueSize<9, 119, true>
    {
        typedef CosValuesTableQ9_119 CosTableType;
    };
    #endif // TINYMIND_USE_COS_9_119

    #if TINYMIND_USE_COS_10_118
    template<>
    struct CosTableValueSize<10, 118, true>
    {
        typedef CosValuesTableQ10_118 CosTableType;
    };
    #endif // TINYMIND_USE_COS_10_118

    #if TINYMIND_USE_COS_11_117
    template<>
    struct CosTableValueSize<11, 117, true>
    {
        typedef CosValuesTableQ11_117 CosTableType;
    };
    #endif // TINYMIND_USE_COS_11_117

    #if TINYMIND_USE_COS_12_116
    template<>
    struct CosTableValueSize<12, 116, true>
    {
        typedef CosValuesTableQ12_116 CosTableType;
    };
    #endif // TINYMIND_USE_COS_12_116

    #if TINYMIND_USE_COS_13_115
    template<>
    struct CosTableValueSize<13, 115, true>
    {
        typedef CosValuesTableQ13_115 CosTableType;
    };
    #endif // TINYMIND_USE_COS_13_115

    #if TINYMIND_USE_COS_14_114
    template<>
    struct CosTableValueSize<14, 114, true>
    {
        typedef CosValuesTableQ14_114 CosTableType;
    };
    #endif // TINYMIND_USE_COS_14_114

    #if TINYMIND_USE_COS_15_113
    template<>
    struct CosTableValueSize<15, 113, true>
    {
        typedef CosValuesTableQ15_113 CosTableType;
    };
    #endif // TINYMIND_USE_COS_15_113

    #if TINYMIND_USE_COS_16_112
    template<>
    struct CosTableValueSize<16, 112, true>
    {
        typedef CosValuesTableQ16_112 CosTableType;
    };
    #endif // TINYMIND_USE_COS_16_112

    #if TINYMIND_USE_COS_17_111
    template<>
    struct CosTableValueSize<17, 111, true>
    {
        typedef CosValuesTableQ17_111 CosTableType;
    };
    #endif // TINYMIND_USE_COS_17_111

    #if TINYMIND_USE_COS_18_110
    template<>
    struct CosTableValueSize<18, 110, true>
    {
        typedef CosValuesTableQ18_110 CosTableType;
    };
    #endif // TINYMIND_USE_COS_18_110

    #if TINYMIND_USE_COS_19_109
    template<>
    struct CosTableValueSize<19, 109, true>
    {
        typedef CosValuesTableQ19_109 CosTableType;
    };
    #endif // TINYMIND_USE_COS_19_109

    #if TINYMIND_USE_COS_20_108
    template<>
    struct CosTableValueSize<20, 108, true>
    {
        typedef CosValuesTableQ20_108 CosTableType;
    };
    #endif // TINYMIND_USE_COS_20_108

    #if TINYMIND_USE_COS_21_107
    template<>
    struct CosTableValueSize<21, 107, true>
    {
        typedef CosValuesTableQ21_107 CosTableType;
    };
    #endif // TINYMIND_USE_COS_21_107

    #if TINYMIND_USE_COS_22_106
    template<>
    struct CosTableValueSize<22, 106, true>
    {
        typedef CosValuesTableQ22_106 CosTableType;
    };
    #endif // TINYMIND_USE_COS_22_106

    #if TINYMIND_USE_COS_23_105
    template<>
    struct CosTableValueSize<23, 105, true>
    {
        typedef CosValuesTableQ23_105 CosTableType;
    };
    #endif // TINYMIND_USE_COS_23_105

    #if TINYMIND_USE_COS_24_104
    template<>
    struct CosTableValueSize<24, 104, true>
    {
        typedef CosValuesTableQ24_104 CosTableType;
    };
    #endif // TINYMIND_USE_COS_24_104

    #if TINYMIND_USE_COS_25_103
    template<>
    struct CosTableValueSize<25, 103, true>
    {
        typedef CosValuesTableQ25_103 CosTableType;
    };
    #endif // TINYMIND_USE_COS_25_103

    #if TINYMIND_USE_COS_26_102
    template<>
    struct CosTableValueSize<26, 102, true>
    {
        typedef CosValuesTableQ26_102 CosTableType;
    };
    #endif // TINYMIND_USE_COS_26_102

    #if TINYMIND_USE_COS_27_101
    template<>
    struct CosTableValueSize<27, 101, true>
    {
        typedef CosValuesTableQ27_101 CosTableType;
    };
    #endif // TINYMIND_USE_COS_27_101

    #if TINYMIND_USE_COS_28_100
    template<>
    struct CosTableValueSize<28, 100, true>
    {
        typedef CosValuesTableQ28_100 CosTableType;
    };
    #endif // TINYMIND_USE_COS_28_100

    #if TINYMIND_USE_COS_29_99
    template<>
    struct CosTableValueSize<29, 99, true>
    {
        typedef CosValuesTableQ29_99 CosTableType;
    };
    #endif // TINYMIND_USE_COS_29_99

    #if TINYMIND_USE_COS_30_98
    template<>
    struct CosTableValueSize<30, 98, true>
    {
        typedef CosValuesTableQ30_98 CosTableType;
    };
    #endif // TINYMIND_USE_COS_30_98

    #if TINYMIND_USE_COS_31_97
    template<>
    struct CosTableValueSize<31, 97, true>
    {
        typedef CosValuesTableQ31_97 CosTableType;
    };
    #endif // TINYMIND_USE_COS_31_97

    #if TINYMIND_USE_COS_32_96
    template<>
    struct CosTableValueSize<32, 96, true>
    {
        typedef CosValuesTableQ32_96 CosTableType;
    };
    #endif // TINYMIND_USE_COS_32_96

    #if TINYMIND_USE_COS_33_95
    template<>
    struct CosTableValueSize<33, 95, true>
    {
        typedef CosValuesTableQ33_95 CosTableType;
    };
    #endif // TINYMIND_USE_COS_33_95

    #if TINYMIND_USE_COS_34_94
    template<>
    struct CosTableValueSize<34, 94, true>
    {
        typedef CosValuesTableQ34_94 CosTableType;
    };
    #endif // TINYMIND_USE_COS_34_94

    #if TINYMIND_USE_COS_35_93
    template<>
    struct CosTableValueSize<35, 93, true>
    {
        typedef CosValuesTableQ35_93 CosTableType;
    };
    #endif // TINYMIND_USE_COS_35_93

    #if TINYMIND_USE_COS_36_92
    template<>
    struct CosTableValueSize<36, 92, true>
    {
        typedef CosValuesTableQ36_92 CosTableType;
    };
    #endif // TINYMIND_USE_COS_36_92

    #if TINYMIND_USE_COS_37_91
    template<>
    struct CosTableValueSize<37, 91, true>
    {
        typedef CosValuesTableQ37_91 CosTableType;
    };
    #endif // TINYMIND_USE_COS_37_91

    #if TINYMIND_USE_COS_38_90
    template<>
    struct CosTableValueSize<38, 90, true>
    {
        typedef CosValuesTableQ38_90 CosTableType;
    };
    #endif // TINYMIND_USE_COS_38_90

    #if TINYMIND_USE_COS_39_89
    template<>
    struct CosTableValueSize<39, 89, true>
    {
        typedef CosValuesTableQ39_89 CosTableType;
    };
    #endif // TINYMIND_USE_COS_39_89

    #if TINYMIND_USE_COS_40_88
    template<>
    struct CosTableValueSize<40, 88, true>
    {
        typedef CosValuesTableQ40_88 CosTableType;
    };
    #endif // TINYMIND_USE_COS_40_88

    #if TINYMIND_USE_COS_41_87
    template<>
    struct CosTableValueSize<41, 87, true>
    {
        typedef CosValuesTableQ41_87 CosTableType;
    };
    #endif // TINYMIND_USE_COS_41_87

    #if TINYMIND_USE_COS_42_86
    template<>
    struct CosTableValueSize<42, 86, true>
    {
        typedef CosValuesTableQ42_86 CosTableType;
    };
    #endif // TINYMIND_USE_COS_42_86

    #if TINYMIND_USE_COS_43_85
    template<>
    struct CosTableValueSize<43, 85, true>
    {
        typedef CosValuesTableQ43_85 CosTableType;
    };
    #endif // TINYMIND_USE_COS_43_85

    #if TINYMIND_USE_COS_44_84
    template<>
    struct CosTableValueSize<44, 84, true>
    {
        typedef CosValuesTableQ44_84 CosTableType;
    };
    #endif // TINYMIND_USE_COS_44_84

    #if TINYMIND_USE_COS_45_83
    template<>
    struct CosTableValueSize<45, 83, true>
    {
        typedef CosValuesTableQ45_83 CosTableType;
    };
    #endif // TINYMIND_USE_COS_45_83

    #if TINYMIND_USE_COS_46_82
    template<>
    struct CosTableValueSize<46, 82, true>
    {
        typedef CosValuesTableQ46_82 CosTableType;
    };
    #endif // TINYMIND_USE_COS_46_82

    #if TINYMIND_USE_COS_47_81
    template<>
    struct CosTableValueSize<47, 81, true>
    {
        typedef CosValuesTableQ47_81 CosTableType;
    };
    #endif // TINYMIND_USE_COS_47_81

    #if TINYMIND_USE_COS_48_80
    template<>
    struct CosTableValueSize<48, 80, true>
    {
        typedef CosValuesTableQ48_80 CosTableType;
    };
    #endif // TINYMIND_USE_COS_48_80

    #if TINYMIND_USE_COS_49_79
    template<>
    struct CosTableValueSize<49, 79, true>
    {
        typedef CosValuesTableQ49_79 CosTableType;
    };
    #endif // TINYMIND_USE_COS_49_79

    #if TINYMIND_USE_COS_50_78
    template<>
    struct CosTableValueSize<50, 78, true>
    {
        typedef CosValuesTableQ50_78 CosTableType;
    };
    #endif // TINYMIND_USE_COS_50_78

    #if TINYMIND_USE_COS_51_77
    template<>
    struct CosTableValueSize<51, 77, true>
    {
        typedef CosValuesTableQ51_77 CosTableType;
    };
    #endif // TINYMIND_USE_COS_51_77

    #if TINYMIND_USE_COS_52_76
    template<>
    struct CosTableValueSize<52, 76, true>
    {
        typedef CosValuesTableQ52_76 CosTableType;
    };
    #endif // TINYMIND_USE_COS_52_76

    #if TINYMIND_USE_COS_53_75
    template<>
    struct CosTableValueSize<53, 75, true>
    {
        typedef CosValuesTableQ53_75 CosTableType;
    };
    #endif // TINYMIND_USE_COS_53_75

    #if TINYMIND_USE_COS_54_74
    template<>
    struct CosTableValueSize<54, 74, true>
    {
        typedef CosValuesTableQ54_74 CosTableType;
    };
    #endif // TINYMIND_USE_COS_54_74

    #if TINYMIND_USE_COS_55_73
    template<>
    struct CosTableValueSize<55, 73, true>
    {
        typedef CosValuesTableQ55_73 CosTableType;
    };
    #endif // TINYMIND_USE_COS_55_73

    #if TINYMIND_USE_COS_56_72
    template<>
    struct CosTableValueSize<56, 72, true>
    {
        typedef CosValuesTableQ56_72 CosTableType;
    };
    #endif // TINYMIND_USE_COS_56_72

    #if TINYMIND_USE_COS_57_71
    template<>
    struct CosTableValueSize<57, 71, true>
    {
        typedef CosValuesTableQ57_71 CosTableType;
    };
    #endif // TINYMIND_USE_COS_57_71

    #if TINYMIND_USE_COS_58_70
    template<>
    struct CosTableValueSize<58, 70, true>
    {
        typedef CosValuesTableQ58_70 CosTableType;
    };
    #endif // TINYMIND_USE_COS_58_70

    #if TINYMIND_USE_COS_59_69
    template<>
    struct CosTableValueSize<59, 69, true>
    {
        typedef CosValuesTableQ59_69 CosTableType;
    };
    #endif // TINYMIND_USE_COS_59_69

    #if TINYMIND_USE_COS_60_68
    template<>
    struct CosTableValueSize<60, 68, true>
    {
        typedef CosValuesTableQ60_68 CosTableType;
    };
    #endif // TINYMIND_USE_COS_60_68

    #if TINYMIND_USE_COS_61_67
    template<>
    struct CosTableValueSize<61, 67, true>
    {
        typedef CosValuesTableQ61_67 CosTableType;
    };
    #endif // TINYMIND_USE_COS_61_67

    #if TINYMIND_USE_COS_62_66
    template<>
    struct CosTableValueSize<62, 66, true>
    {
        typedef CosValuesTableQ62_66 CosTableType;
    };
    #endif // TINYMIND_USE_COS_62_66

    #if TINYMIND_USE_COS_63_65
    template<>
    struct CosTableValueSize<63, 65, true>
    {
        typedef CosValuesTableQ63_65 CosTableType;
    };
    #endif // TINYMIND_USE_COS_63_65

    #if TINYMIND_USE_COS_64_64
    template<>
    struct CosTableValueSize<64, 64, true>
    {
        typedef CosValuesTableQ64_64 CosTableType;
    };
    #endif // TINYMIND_USE_COS_64_64

    #if TINYMIND_USE_COS_65_63
    template<>
    struct CosTableValueSize<65, 63, true>
    {
        typedef CosValuesTableQ65_63 CosTableType;
    };
    #endif // TINYMIND_USE_COS_65_63

    #if TINYMIND_USE_COS_66_62
    template<>
    struct CosTableValueSize<66, 62, true>
    {
        typedef CosValuesTableQ66_62 CosTableType;
    };
    #endif // TINYMIND_USE_COS_66_62

    #if TINYMIND_USE_COS_67_61
    template<>
    struct CosTableValueSize<67, 61, true>
    {
        typedef CosValuesTableQ67_61 CosTableType;
    };
    #endif // TINYMIND_USE_COS_67_61

    #if TINYMIND_USE_COS_68_60
    template<>
    struct CosTableValueSize<68, 60, true>
    {
        typedef CosValuesTableQ68_60 CosTableType;
    };
    #endif // TINYMIND_USE_COS_68_60

    #if TINYMIND_USE_COS_69_59
    template<>
    struct CosTableValueSize<69, 59, true>
    {
        typedef CosValuesTableQ69_59 CosTableType;
    };
    #endif // TINYMIND_USE_COS_69_59

    #if TINYMIND_USE_COS_70_58
    template<>
    struct CosTableValueSize<70, 58, true>
    {
        typedef CosValuesTableQ70_58 CosTableType;
    };
    #endif // TINYMIND_USE_COS_70_58

    #if TINYMIND_USE_COS_71_57
    template<>
    struct CosTableValueSize<71, 57, true>
    {
        typedef CosValuesTableQ71_57 CosTableType;
    };
    #endif // TINYMIND_USE_COS_71_57

    #if TINYMIND_USE_COS_72_56
    template<>
    struct CosTableValueSize<72, 56, true>
    {
        typedef CosValuesTableQ72_56 CosTableType;
    };
    #endif // TINYMIND_USE_COS_72_56

    #if TINYMIND_USE_COS_73_55
    template<>
    struct CosTableValueSize<73, 55, true>
    {
        typedef CosValuesTableQ73_55 CosTableType;
    };
    #endif // TINYMIND_USE_COS_73_55

    #if TINYMIND_USE_COS_74_54
    template<>
    struct CosTableValueSize<74, 54, true>
    {
        typedef CosValuesTableQ74_54 CosTableType;
    };
    #endif // TINYMIND_USE_COS_74_54

    #if TINYMIND_USE_COS_75_53
    template<>
    struct CosTableValueSize<75, 53, true>
    {
        typedef CosValuesTableQ75_53 CosTableType;
    };
    #endif // TINYMIND_USE_COS_75_53

    #if TINYMIND_USE_COS_76_52
    template<>
    struct CosTableValueSize<76, 52, true>
    {
        typedef CosValuesTableQ76_52 CosTableType;
    };
    #endif // TINYMIND_USE_COS_76_52

    #if TINYMIND_USE_COS_77_51
    template<>
    struct CosTableValueSize<77, 51, true>
    {
        typedef CosValuesTableQ77_51 CosTableType;
    };
    #endif // TINYMIND_USE_COS_77_51

    #if TINYMIND_USE_COS_78_50
    template<>
    struct CosTableValueSize<78, 50, true>
    {
        typedef CosValuesTableQ78_50 CosTableType;
    };
    #endif // TINYMIND_USE_COS_78_50

    #if TINYMIND_USE_COS_79_49
    template<>
    struct CosTableValueSize<79, 49, true>
    {
        typedef CosValuesTableQ79_49 CosTableType;
    };
    #endif // TINYMIND_USE_COS_79_49

    #if TINYMIND_USE_COS_80_48
    template<>
    struct CosTableValueSize<80, 48, true>
    {
        typedef CosValuesTableQ80_48 CosTableType;
    };
    #endif // TINYMIND_USE_COS_80_48

    #if TINYMIND_USE_COS_81_47
    template<>
    struct CosTableValueSize<81, 47, true>
    {
        typedef CosValuesTableQ81_47 CosTableType;
    };
    #endif // TINYMIND_USE_COS_81_47

    #if TINYMIND_USE_COS_82_46
    template<>
    struct CosTableValueSize<82, 46, true>
    {
        typedef CosValuesTableQ82_46 CosTableType;
    };
    #endif // TINYMIND_USE_COS_82_46

    #if TINYMIND_USE_COS_83_45
    template<>
    struct CosTableValueSize<83, 45, true>
    {
        typedef CosValuesTableQ83_45 CosTableType;
    };
    #endif // TINYMIND_USE_COS_83_45

    #if TINYMIND_USE_COS_84_44
    template<>
    struct CosTableValueSize<84, 44, true>
    {
        typedef CosValuesTableQ84_44 CosTableType;
    };
    #endif // TINYMIND_USE_COS_84_44

    #if TINYMIND_USE_COS_85_43
    template<>
    struct CosTableValueSize<85, 43, true>
    {
        typedef CosValuesTableQ85_43 CosTableType;
    };
    #endif // TINYMIND_USE_COS_85_43

    #if TINYMIND_USE_COS_86_42
    template<>
    struct CosTableValueSize<86, 42, true>
    {
        typedef CosValuesTableQ86_42 CosTableType;
    };
    #endif // TINYMIND_USE_COS_86_42

    #if TINYMIND_USE_COS_87_41
    template<>
    struct CosTableValueSize<87, 41, true>
    {
        typedef CosValuesTableQ87_41 CosTableType;
    };
    #endif // TINYMIND_USE_COS_87_41

    #if TINYMIND_USE_COS_88_40
    template<>
    struct CosTableValueSize<88, 40, true>
    {
        typedef CosValuesTableQ88_40 CosTableType;
    };
    #endif // TINYMIND_USE_COS_88_40

    #if TINYMIND_USE_COS_89_39
    template<>
    struct CosTableValueSize<89, 39, true>
    {
        typedef CosValuesTableQ89_39 CosTableType;
    };
    #endif // TINYMIND_USE_COS_89_39

    #if TINYMIND_USE_COS_90_38
    template<>
    struct CosTableValueSize<90, 38, true>
    {
        typedef CosValuesTableQ90_38 CosTableType;
    };
    #endif // TINYMIND_USE_COS_90_38

    #if TINYMIND_USE_COS_91_37
    template<>
    struct CosTableValueSize<91, 37, true>
    {
        typedef CosValuesTableQ91_37 CosTableType;
    };
    #endif // TINYMIND_USE_COS_91_37

    #if TINYMIND_USE_COS_92_36
    template<>
    struct CosTableValueSize<92, 36, true>
    {
        typedef CosValuesTableQ92_36 CosTableType;
    };
    #endif // TINYMIND_USE_COS_92_36

    #if TINYMIND_USE_COS_93_35
    template<>
    struct CosTableValueSize<93, 35, true>
    {
        typedef CosValuesTableQ93_35 CosTableType;
    };
    #endif // TINYMIND_USE_COS_93_35

    #if TINYMIND_USE_COS_94_34
    template<>
    struct CosTableValueSize<94, 34, true>
    {
        typedef CosValuesTableQ94_34 CosTableType;
    };
    #endif // TINYMIND_USE_COS_94_34

    #if TINYMIND_USE_COS_95_33
    template<>
    struct CosTableValueSize<95, 33, true>
    {
        typedef CosValuesTableQ95_33 CosTableType;
    };
    #endif // TINYMIND_USE_COS_95_33

    #if TINYMIND_USE_COS_96_32
    template<>
    struct CosTableValueSize<96, 32, true>
    {
        typedef CosValuesTableQ96_32 CosTableType;
    };
    #endif // TINYMIND_USE_COS_96_32

    #if TINYMIND_USE_COS_97_31
    template<>
    struct CosTableValueSize<97, 31, true>
    {
        typedef CosValuesTableQ97_31 CosTableType;
    };
    #endif // TINYMIND_USE_COS_97_31

    #if TINYMIND_USE_COS_98_30
    template<>
    struct CosTableValueSize<98, 30, true>
    {
        typedef CosValuesTableQ98_30 CosTableType;
    };
    #endif // TINYMIND_USE_COS_98_30

    #if TINYMIND_USE_COS_99_29
    template<>
    struct CosTableValueSize<99, 29, true>
    {
        typedef CosValuesTableQ99_29 CosTableType;
    };
    #endif // TINYMIND_USE_COS_99_29

    #if TINYMIND_USE_COS_100_28
    template<>
    struct CosTableValueSize<100, 28, true>
    {
        typedef CosValuesTableQ100_28 CosTableType;
    };
    #endif // TINYMIND_USE_COS_100_28

    #if TINYMIND_USE_COS_101_27
    template<>
    struct CosTableValueSize<101, 27, true>
    {
        typedef CosValuesTableQ101_27 CosTableType;
    };
    #endif // TINYMIND_USE_COS_101_27

    #if TINYMIND_USE_COS_102_26
    template<>
    struct CosTableValueSize<102, 26, true>
    {
        typedef CosValuesTableQ102_26 CosTableType;
    };
    #endif // TINYMIND_USE_COS_102_26

    #if TINYMIND_USE_COS_103_25
    template<>
    struct CosTableValueSize<103, 25, true>
    {
        typedef CosValuesTableQ103_25 CosTableType;
    };
    #endif // TINYMIND_USE_COS_103_25

    #if TINYMIND_USE_COS_104_24
    template<>
    struct CosTableValueSize<104, 24, true>
    {
        typedef CosValuesTableQ104_24 CosTableType;
    };
    #endif // TINYMIND_USE_COS_104_24

    #if TINYMIND_USE_COS_105_23
    template<>
    struct CosTableValueSize<105, 23, true>
    {
        typedef CosValuesTableQ105_23 CosTableType;
    };
    #endif // TINYMIND_USE_COS_105_23

    #if TINYMIND_USE_COS_106_22
    template<>
    struct CosTableValueSize<106, 22, true>
    {
        typedef CosValuesTableQ106_22 CosTableType;
    };
    #endif // TINYMIND_USE_COS_106_22

    #if TINYMIND_USE_COS_107_21
    template<>
    struct CosTableValueSize<107, 21, true>
    {
        typedef CosValuesTableQ107_21 CosTableType;
    };
    #endif // TINYMIND_USE_COS_107_21

    #if TINYMIND_USE_COS_108_20
    template<>
    struct CosTableValueSize<108, 20, true>
    {
        typedef CosValuesTableQ108_20 CosTableType;
    };
    #endif // TINYMIND_USE_COS_108_20

    #if TINYMIND_USE_COS_109_19
    template<>
    struct CosTableValueSize<109, 19, true>
    {
        typedef CosValuesTableQ109_19 CosTableType;
    };
    #endif // TINYMIND_USE_COS_109_19

    #if TINYMIND_USE_COS_110_18
    template<>
    struct CosTableValueSize<110, 18, true>
    {
        typedef CosValuesTableQ110_18 CosTableType;
    };
    #endif // TINYMIND_USE_COS_110_18

    #if TINYMIND_USE_COS_111_17
    template<>
    struct CosTableValueSize<111, 17, true>
    {
        typedef CosValuesTableQ111_17 CosTableType;
    };
    #endif // TINYMIND_USE_COS_111_17

    #if TINYMIND_USE_COS_112_16
    template<>
    struct CosTableValueSize<112, 16, true>
    {
        typedef CosValuesTableQ112_16 CosTableType;
    };
    #endif // TINYMIND_USE_COS_112_16

    #if TINYMIND_USE_COS_113_15
    template<>
    struct CosTableValueSize<113, 15, true>
    {
        typedef CosValuesTableQ113_15 CosTableType;
    };
    #endif // TINYMIND_USE_COS_113_15

    #if TINYMIND_USE_COS_114_14
    template<>
    struct CosTableValueSize<114, 14, true>
    {
        typedef CosValuesTableQ114_14 CosTableType;
    };
    #endif // TINYMIND_USE_COS_114_14

    #if TINYMIND_USE_COS_115_13
    template<>
    struct CosTableValueSize<115, 13, true>
    {
        typedef CosValuesTableQ115_13 CosTableType;
    };
    #endif // TINYMIND_USE_COS_115_13

    #if TINYMIND_USE_COS_116_12
    template<>
    struct CosTableValueSize<116, 12, true>
    {
        typedef CosValuesTableQ116_12 CosTableType;
    };
    #endif // TINYMIND_USE_COS_116_12

    #if TINYMIND_USE_COS_117_11
    template<>
    struct CosTableValueSize<117, 11, true>
    {
        typedef CosValuesTableQ117_11 CosTableType;
    };
    #endif // TINYMIND_USE_COS_117_11

    #if TINYMIND_USE_COS_118_10
    template<>
    struct CosTableValueSize<118, 10, true>
    {
        typedef CosValuesTableQ118_10 CosTableType;
    };
    #endif // TINYMIND_USE_COS_118_10

    #if TINYMIND_USE_COS_119_9
    template<>
    struct CosTableValueSize<119, 9, true>
    {
        typedef CosValuesTableQ119_9 CosTableType;
    };
    #endif // TINYMIND_USE_COS_119_9

    #if TINYMIND_USE_COS_120_8
    template<>
    struct CosTableValueSize<120, 8, true>
    {
        typedef CosValuesTableQ120_8 CosTableType;
    };
    #endif // TINYMIND_USE_COS_120_8

    #if TINYMIND_USE_COS_121_7
    template<>
    struct CosTableValueSize<121, 7, true>
    {
        typedef CosValuesTableQ121_7 CosTableType;
    };
    #endif // TINYMIND_USE_COS_121_7

    #if TINYMIND_USE_COS_122_6
    template<>
    struct CosTableValueSize<122, 6, true>
    {
        typedef CosValuesTableQ122_6 CosTableType;
    };
    #endif // TINYMIND_USE_COS_122_6

    #if TINYMIND_USE_COS_123_5
    template<>
    struct CosTableValueSize<123, 5, true>
    {
        typedef CosValuesTableQ123_5 CosTableType;
    };
    #endif // TINYMIND_USE_COS_123_5

    #if TINYMIND_USE_COS_124_4
    template<>
    struct CosTableValueSize<124, 4, true>
    {
        typedef CosValuesTableQ124_4 CosTableType;
    };
    #endif // TINYMIND_USE_COS_124_4

    #if TINYMIND_USE_COS_125_3
    template<>
    struct CosTableValueSize<125, 3, true>
    {
        typedef CosValuesTableQ125_3 CosTableType;
    };
    #endif // TINYMIND_USE_COS_125_3

    #if TINYMIND_USE_COS_126_2
    template<>
    struct CosTableValueSize<126, 2, true>
    {
        typedef CosValuesTableQ126_2 CosTableType;
    };
    #endif // TINYMIND_USE_COS_126_2

    #if TINYMIND_USE_COS_127_1
    template<>
    struct CosTableValueSize<127, 1, true>
    {
        typedef CosValuesTableQ127_1 CosTableType;
    };
    #endif // TINYMIND_USE_COS_127_1

    template<unsigned FixedBits,unsigned FracBits, bool IsSigned>
    struct CosValuesTableSelector
    {
        typedef typename CosTableValueSize<FixedBits, FracBits, IsSigned>::CosTableType CosTableType;
    };
}
