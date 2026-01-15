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

#include "tanhValues8Bit.hpp"
#include "tanhValues16Bit.hpp"
#include "tanhValues32Bit.hpp"
#include "tanhValues64Bit.hpp"
#include "tanhValues128Bit.hpp"

namespace tinymind {
    template<unsigned FixedBits, unsigned FracBits, bool IsSigned>
    struct TanhTableValueSize
    {
    };

    #if TINYMIND_USE_TANH_1_7
    template<>
    struct TanhTableValueSize<1, 7, true>
    {
        typedef TanhValuesTableQ1_7 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_1_7

    #if TINYMIND_USE_TANH_2_6
    template<>
    struct TanhTableValueSize<2, 6, true>
    {
        typedef TanhValuesTableQ2_6 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_2_6

    #if TINYMIND_USE_TANH_3_5
    template<>
    struct TanhTableValueSize<3, 5, true>
    {
        typedef TanhValuesTableQ3_5 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_3_5

    #if TINYMIND_USE_TANH_4_4
    template<>
    struct TanhTableValueSize<4, 4, true>
    {
        typedef TanhValuesTableQ4_4 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_4_4

    #if TINYMIND_USE_TANH_5_3
    template<>
    struct TanhTableValueSize<5, 3, true>
    {
        typedef TanhValuesTableQ5_3 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_5_3

    #if TINYMIND_USE_TANH_6_2
    template<>
    struct TanhTableValueSize<6, 2, true>
    {
        typedef TanhValuesTableQ6_2 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_6_2

    #if TINYMIND_USE_TANH_7_1
    template<>
    struct TanhTableValueSize<7, 1, true>
    {
        typedef TanhValuesTableQ7_1 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_7_1

    #if TINYMIND_USE_TANH_1_15
    template<>
    struct TanhTableValueSize<1, 15, true>
    {
        typedef TanhValuesTableQ1_15 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_1_15

    #if TINYMIND_USE_TANH_2_14
    template<>
    struct TanhTableValueSize<2, 14, true>
    {
        typedef TanhValuesTableQ2_14 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_2_14

    #if TINYMIND_USE_TANH_3_13
    template<>
    struct TanhTableValueSize<3, 13, true>
    {
        typedef TanhValuesTableQ3_13 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_3_13

    #if TINYMIND_USE_TANH_4_12
    template<>
    struct TanhTableValueSize<4, 12, true>
    {
        typedef TanhValuesTableQ4_12 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_4_12

    #if TINYMIND_USE_TANH_5_11
    template<>
    struct TanhTableValueSize<5, 11, true>
    {
        typedef TanhValuesTableQ5_11 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_5_11

    #if TINYMIND_USE_TANH_6_10
    template<>
    struct TanhTableValueSize<6, 10, true>
    {
        typedef TanhValuesTableQ6_10 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_6_10

    #if TINYMIND_USE_TANH_7_9
    template<>
    struct TanhTableValueSize<7, 9, true>
    {
        typedef TanhValuesTableQ7_9 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_7_9

    #if TINYMIND_USE_TANH_8_8
    template<>
    struct TanhTableValueSize<8, 8, true>
    {
        typedef TanhValuesTableQ8_8 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_8_8

    #if TINYMIND_USE_TANH_9_7
    template<>
    struct TanhTableValueSize<9, 7, true>
    {
        typedef TanhValuesTableQ9_7 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_9_7

    #if TINYMIND_USE_TANH_10_6
    template<>
    struct TanhTableValueSize<10, 6, true>
    {
        typedef TanhValuesTableQ10_6 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_10_6

    #if TINYMIND_USE_TANH_11_5
    template<>
    struct TanhTableValueSize<11, 5, true>
    {
        typedef TanhValuesTableQ11_5 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_11_5

    #if TINYMIND_USE_TANH_12_4
    template<>
    struct TanhTableValueSize<12, 4, true>
    {
        typedef TanhValuesTableQ12_4 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_12_4

    #if TINYMIND_USE_TANH_13_3
    template<>
    struct TanhTableValueSize<13, 3, true>
    {
        typedef TanhValuesTableQ13_3 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_13_3

    #if TINYMIND_USE_TANH_14_2
    template<>
    struct TanhTableValueSize<14, 2, true>
    {
        typedef TanhValuesTableQ14_2 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_14_2

    #if TINYMIND_USE_TANH_15_1
    template<>
    struct TanhTableValueSize<15, 1, true>
    {
        typedef TanhValuesTableQ15_1 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_15_1

    #if TINYMIND_USE_TANH_1_31
    template<>
    struct TanhTableValueSize<1, 31, true>
    {
        typedef TanhValuesTableQ1_31 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_1_31

    #if TINYMIND_USE_TANH_2_30
    template<>
    struct TanhTableValueSize<2, 30, true>
    {
        typedef TanhValuesTableQ2_30 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_2_30

    #if TINYMIND_USE_TANH_3_29
    template<>
    struct TanhTableValueSize<3, 29, true>
    {
        typedef TanhValuesTableQ3_29 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_3_29

    #if TINYMIND_USE_TANH_4_28
    template<>
    struct TanhTableValueSize<4, 28, true>
    {
        typedef TanhValuesTableQ4_28 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_4_28

    #if TINYMIND_USE_TANH_5_27
    template<>
    struct TanhTableValueSize<5, 27, true>
    {
        typedef TanhValuesTableQ5_27 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_5_27

    #if TINYMIND_USE_TANH_6_26
    template<>
    struct TanhTableValueSize<6, 26, true>
    {
        typedef TanhValuesTableQ6_26 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_6_26

    #if TINYMIND_USE_TANH_7_25
    template<>
    struct TanhTableValueSize<7, 25, true>
    {
        typedef TanhValuesTableQ7_25 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_7_25

    #if TINYMIND_USE_TANH_8_24
    template<>
    struct TanhTableValueSize<8, 24, true>
    {
        typedef TanhValuesTableQ8_24 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_8_24

    #if TINYMIND_USE_TANH_9_23
    template<>
    struct TanhTableValueSize<9, 23, true>
    {
        typedef TanhValuesTableQ9_23 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_9_23

    #if TINYMIND_USE_TANH_10_22
    template<>
    struct TanhTableValueSize<10, 22, true>
    {
        typedef TanhValuesTableQ10_22 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_10_22

    #if TINYMIND_USE_TANH_11_21
    template<>
    struct TanhTableValueSize<11, 21, true>
    {
        typedef TanhValuesTableQ11_21 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_11_21

    #if TINYMIND_USE_TANH_12_20
    template<>
    struct TanhTableValueSize<12, 20, true>
    {
        typedef TanhValuesTableQ12_20 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_12_20

    #if TINYMIND_USE_TANH_13_19
    template<>
    struct TanhTableValueSize<13, 19, true>
    {
        typedef TanhValuesTableQ13_19 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_13_19

    #if TINYMIND_USE_TANH_14_18
    template<>
    struct TanhTableValueSize<14, 18, true>
    {
        typedef TanhValuesTableQ14_18 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_14_18

    #if TINYMIND_USE_TANH_15_17
    template<>
    struct TanhTableValueSize<15, 17, true>
    {
        typedef TanhValuesTableQ15_17 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_15_17

    #if TINYMIND_USE_TANH_16_16
    template<>
    struct TanhTableValueSize<16, 16, true>
    {
        typedef TanhValuesTableQ16_16 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_16_16

    #if TINYMIND_USE_TANH_17_15
    template<>
    struct TanhTableValueSize<17, 15, true>
    {
        typedef TanhValuesTableQ17_15 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_17_15

    #if TINYMIND_USE_TANH_18_14
    template<>
    struct TanhTableValueSize<18, 14, true>
    {
        typedef TanhValuesTableQ18_14 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_18_14

    #if TINYMIND_USE_TANH_19_13
    template<>
    struct TanhTableValueSize<19, 13, true>
    {
        typedef TanhValuesTableQ19_13 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_19_13

    #if TINYMIND_USE_TANH_20_12
    template<>
    struct TanhTableValueSize<20, 12, true>
    {
        typedef TanhValuesTableQ20_12 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_20_12

    #if TINYMIND_USE_TANH_21_11
    template<>
    struct TanhTableValueSize<21, 11, true>
    {
        typedef TanhValuesTableQ21_11 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_21_11

    #if TINYMIND_USE_TANH_22_10
    template<>
    struct TanhTableValueSize<22, 10, true>
    {
        typedef TanhValuesTableQ22_10 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_22_10

    #if TINYMIND_USE_TANH_23_9
    template<>
    struct TanhTableValueSize<23, 9, true>
    {
        typedef TanhValuesTableQ23_9 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_23_9

    #if TINYMIND_USE_TANH_24_8
    template<>
    struct TanhTableValueSize<24, 8, true>
    {
        typedef TanhValuesTableQ24_8 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_24_8

    #if TINYMIND_USE_TANH_25_7
    template<>
    struct TanhTableValueSize<25, 7, true>
    {
        typedef TanhValuesTableQ25_7 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_25_7

    #if TINYMIND_USE_TANH_26_6
    template<>
    struct TanhTableValueSize<26, 6, true>
    {
        typedef TanhValuesTableQ26_6 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_26_6

    #if TINYMIND_USE_TANH_27_5
    template<>
    struct TanhTableValueSize<27, 5, true>
    {
        typedef TanhValuesTableQ27_5 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_27_5

    #if TINYMIND_USE_TANH_28_4
    template<>
    struct TanhTableValueSize<28, 4, true>
    {
        typedef TanhValuesTableQ28_4 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_28_4

    #if TINYMIND_USE_TANH_29_3
    template<>
    struct TanhTableValueSize<29, 3, true>
    {
        typedef TanhValuesTableQ29_3 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_29_3

    #if TINYMIND_USE_TANH_30_2
    template<>
    struct TanhTableValueSize<30, 2, true>
    {
        typedef TanhValuesTableQ30_2 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_30_2

    #if TINYMIND_USE_TANH_31_1
    template<>
    struct TanhTableValueSize<31, 1, true>
    {
        typedef TanhValuesTableQ31_1 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_31_1

    #if TINYMIND_USE_TANH_1_63
    template<>
    struct TanhTableValueSize<1, 63, true>
    {
        typedef TanhValuesTableQ1_63 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_1_63

    #if TINYMIND_USE_TANH_2_62
    template<>
    struct TanhTableValueSize<2, 62, true>
    {
        typedef TanhValuesTableQ2_62 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_2_62

    #if TINYMIND_USE_TANH_3_61
    template<>
    struct TanhTableValueSize<3, 61, true>
    {
        typedef TanhValuesTableQ3_61 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_3_61

    #if TINYMIND_USE_TANH_4_60
    template<>
    struct TanhTableValueSize<4, 60, true>
    {
        typedef TanhValuesTableQ4_60 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_4_60

    #if TINYMIND_USE_TANH_5_59
    template<>
    struct TanhTableValueSize<5, 59, true>
    {
        typedef TanhValuesTableQ5_59 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_5_59

    #if TINYMIND_USE_TANH_6_58
    template<>
    struct TanhTableValueSize<6, 58, true>
    {
        typedef TanhValuesTableQ6_58 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_6_58

    #if TINYMIND_USE_TANH_7_57
    template<>
    struct TanhTableValueSize<7, 57, true>
    {
        typedef TanhValuesTableQ7_57 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_7_57

    #if TINYMIND_USE_TANH_8_56
    template<>
    struct TanhTableValueSize<8, 56, true>
    {
        typedef TanhValuesTableQ8_56 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_8_56

    #if TINYMIND_USE_TANH_9_55
    template<>
    struct TanhTableValueSize<9, 55, true>
    {
        typedef TanhValuesTableQ9_55 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_9_55

    #if TINYMIND_USE_TANH_10_54
    template<>
    struct TanhTableValueSize<10, 54, true>
    {
        typedef TanhValuesTableQ10_54 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_10_54

    #if TINYMIND_USE_TANH_11_53
    template<>
    struct TanhTableValueSize<11, 53, true>
    {
        typedef TanhValuesTableQ11_53 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_11_53

    #if TINYMIND_USE_TANH_12_52
    template<>
    struct TanhTableValueSize<12, 52, true>
    {
        typedef TanhValuesTableQ12_52 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_12_52

    #if TINYMIND_USE_TANH_13_51
    template<>
    struct TanhTableValueSize<13, 51, true>
    {
        typedef TanhValuesTableQ13_51 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_13_51

    #if TINYMIND_USE_TANH_14_50
    template<>
    struct TanhTableValueSize<14, 50, true>
    {
        typedef TanhValuesTableQ14_50 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_14_50

    #if TINYMIND_USE_TANH_15_49
    template<>
    struct TanhTableValueSize<15, 49, true>
    {
        typedef TanhValuesTableQ15_49 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_15_49

    #if TINYMIND_USE_TANH_16_48
    template<>
    struct TanhTableValueSize<16, 48, true>
    {
        typedef TanhValuesTableQ16_48 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_16_48

    #if TINYMIND_USE_TANH_17_47
    template<>
    struct TanhTableValueSize<17, 47, true>
    {
        typedef TanhValuesTableQ17_47 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_17_47

    #if TINYMIND_USE_TANH_18_46
    template<>
    struct TanhTableValueSize<18, 46, true>
    {
        typedef TanhValuesTableQ18_46 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_18_46

    #if TINYMIND_USE_TANH_19_45
    template<>
    struct TanhTableValueSize<19, 45, true>
    {
        typedef TanhValuesTableQ19_45 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_19_45

    #if TINYMIND_USE_TANH_20_44
    template<>
    struct TanhTableValueSize<20, 44, true>
    {
        typedef TanhValuesTableQ20_44 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_20_44

    #if TINYMIND_USE_TANH_21_43
    template<>
    struct TanhTableValueSize<21, 43, true>
    {
        typedef TanhValuesTableQ21_43 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_21_43

    #if TINYMIND_USE_TANH_22_42
    template<>
    struct TanhTableValueSize<22, 42, true>
    {
        typedef TanhValuesTableQ22_42 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_22_42

    #if TINYMIND_USE_TANH_23_41
    template<>
    struct TanhTableValueSize<23, 41, true>
    {
        typedef TanhValuesTableQ23_41 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_23_41

    #if TINYMIND_USE_TANH_24_40
    template<>
    struct TanhTableValueSize<24, 40, true>
    {
        typedef TanhValuesTableQ24_40 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_24_40

    #if TINYMIND_USE_TANH_25_39
    template<>
    struct TanhTableValueSize<25, 39, true>
    {
        typedef TanhValuesTableQ25_39 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_25_39

    #if TINYMIND_USE_TANH_26_38
    template<>
    struct TanhTableValueSize<26, 38, true>
    {
        typedef TanhValuesTableQ26_38 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_26_38

    #if TINYMIND_USE_TANH_27_37
    template<>
    struct TanhTableValueSize<27, 37, true>
    {
        typedef TanhValuesTableQ27_37 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_27_37

    #if TINYMIND_USE_TANH_28_36
    template<>
    struct TanhTableValueSize<28, 36, true>
    {
        typedef TanhValuesTableQ28_36 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_28_36

    #if TINYMIND_USE_TANH_29_35
    template<>
    struct TanhTableValueSize<29, 35, true>
    {
        typedef TanhValuesTableQ29_35 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_29_35

    #if TINYMIND_USE_TANH_30_34
    template<>
    struct TanhTableValueSize<30, 34, true>
    {
        typedef TanhValuesTableQ30_34 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_30_34

    #if TINYMIND_USE_TANH_31_33
    template<>
    struct TanhTableValueSize<31, 33, true>
    {
        typedef TanhValuesTableQ31_33 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_31_33

    #if TINYMIND_USE_TANH_32_32
    template<>
    struct TanhTableValueSize<32, 32, true>
    {
        typedef TanhValuesTableQ32_32 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_32_32

    #if TINYMIND_USE_TANH_33_31
    template<>
    struct TanhTableValueSize<33, 31, true>
    {
        typedef TanhValuesTableQ33_31 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_33_31

    #if TINYMIND_USE_TANH_34_30
    template<>
    struct TanhTableValueSize<34, 30, true>
    {
        typedef TanhValuesTableQ34_30 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_34_30

    #if TINYMIND_USE_TANH_35_29
    template<>
    struct TanhTableValueSize<35, 29, true>
    {
        typedef TanhValuesTableQ35_29 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_35_29

    #if TINYMIND_USE_TANH_36_28
    template<>
    struct TanhTableValueSize<36, 28, true>
    {
        typedef TanhValuesTableQ36_28 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_36_28

    #if TINYMIND_USE_TANH_37_27
    template<>
    struct TanhTableValueSize<37, 27, true>
    {
        typedef TanhValuesTableQ37_27 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_37_27

    #if TINYMIND_USE_TANH_38_26
    template<>
    struct TanhTableValueSize<38, 26, true>
    {
        typedef TanhValuesTableQ38_26 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_38_26

    #if TINYMIND_USE_TANH_39_25
    template<>
    struct TanhTableValueSize<39, 25, true>
    {
        typedef TanhValuesTableQ39_25 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_39_25

    #if TINYMIND_USE_TANH_40_24
    template<>
    struct TanhTableValueSize<40, 24, true>
    {
        typedef TanhValuesTableQ40_24 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_40_24

    #if TINYMIND_USE_TANH_41_23
    template<>
    struct TanhTableValueSize<41, 23, true>
    {
        typedef TanhValuesTableQ41_23 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_41_23

    #if TINYMIND_USE_TANH_42_22
    template<>
    struct TanhTableValueSize<42, 22, true>
    {
        typedef TanhValuesTableQ42_22 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_42_22

    #if TINYMIND_USE_TANH_43_21
    template<>
    struct TanhTableValueSize<43, 21, true>
    {
        typedef TanhValuesTableQ43_21 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_43_21

    #if TINYMIND_USE_TANH_44_20
    template<>
    struct TanhTableValueSize<44, 20, true>
    {
        typedef TanhValuesTableQ44_20 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_44_20

    #if TINYMIND_USE_TANH_45_19
    template<>
    struct TanhTableValueSize<45, 19, true>
    {
        typedef TanhValuesTableQ45_19 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_45_19

    #if TINYMIND_USE_TANH_46_18
    template<>
    struct TanhTableValueSize<46, 18, true>
    {
        typedef TanhValuesTableQ46_18 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_46_18

    #if TINYMIND_USE_TANH_47_17
    template<>
    struct TanhTableValueSize<47, 17, true>
    {
        typedef TanhValuesTableQ47_17 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_47_17

    #if TINYMIND_USE_TANH_48_16
    template<>
    struct TanhTableValueSize<48, 16, true>
    {
        typedef TanhValuesTableQ48_16 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_48_16

    #if TINYMIND_USE_TANH_49_15
    template<>
    struct TanhTableValueSize<49, 15, true>
    {
        typedef TanhValuesTableQ49_15 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_49_15

    #if TINYMIND_USE_TANH_50_14
    template<>
    struct TanhTableValueSize<50, 14, true>
    {
        typedef TanhValuesTableQ50_14 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_50_14

    #if TINYMIND_USE_TANH_51_13
    template<>
    struct TanhTableValueSize<51, 13, true>
    {
        typedef TanhValuesTableQ51_13 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_51_13

    #if TINYMIND_USE_TANH_52_12
    template<>
    struct TanhTableValueSize<52, 12, true>
    {
        typedef TanhValuesTableQ52_12 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_52_12

    #if TINYMIND_USE_TANH_53_11
    template<>
    struct TanhTableValueSize<53, 11, true>
    {
        typedef TanhValuesTableQ53_11 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_53_11

    #if TINYMIND_USE_TANH_54_10
    template<>
    struct TanhTableValueSize<54, 10, true>
    {
        typedef TanhValuesTableQ54_10 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_54_10

    #if TINYMIND_USE_TANH_55_9
    template<>
    struct TanhTableValueSize<55, 9, true>
    {
        typedef TanhValuesTableQ55_9 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_55_9

    #if TINYMIND_USE_TANH_56_8
    template<>
    struct TanhTableValueSize<56, 8, true>
    {
        typedef TanhValuesTableQ56_8 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_56_8

    #if TINYMIND_USE_TANH_57_7
    template<>
    struct TanhTableValueSize<57, 7, true>
    {
        typedef TanhValuesTableQ57_7 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_57_7

    #if TINYMIND_USE_TANH_58_6
    template<>
    struct TanhTableValueSize<58, 6, true>
    {
        typedef TanhValuesTableQ58_6 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_58_6

    #if TINYMIND_USE_TANH_59_5
    template<>
    struct TanhTableValueSize<59, 5, true>
    {
        typedef TanhValuesTableQ59_5 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_59_5

    #if TINYMIND_USE_TANH_60_4
    template<>
    struct TanhTableValueSize<60, 4, true>
    {
        typedef TanhValuesTableQ60_4 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_60_4

    #if TINYMIND_USE_TANH_61_3
    template<>
    struct TanhTableValueSize<61, 3, true>
    {
        typedef TanhValuesTableQ61_3 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_61_3

    #if TINYMIND_USE_TANH_62_2
    template<>
    struct TanhTableValueSize<62, 2, true>
    {
        typedef TanhValuesTableQ62_2 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_62_2

    #if TINYMIND_USE_TANH_63_1
    template<>
    struct TanhTableValueSize<63, 1, true>
    {
        typedef TanhValuesTableQ63_1 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_63_1

    #if TINYMIND_USE_TANH_1_127
    template<>
    struct TanhTableValueSize<1, 127, true>
    {
        typedef TanhValuesTableQ1_127 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_1_127

    #if TINYMIND_USE_TANH_2_126
    template<>
    struct TanhTableValueSize<2, 126, true>
    {
        typedef TanhValuesTableQ2_126 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_2_126

    #if TINYMIND_USE_TANH_3_125
    template<>
    struct TanhTableValueSize<3, 125, true>
    {
        typedef TanhValuesTableQ3_125 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_3_125

    #if TINYMIND_USE_TANH_4_124
    template<>
    struct TanhTableValueSize<4, 124, true>
    {
        typedef TanhValuesTableQ4_124 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_4_124

    #if TINYMIND_USE_TANH_5_123
    template<>
    struct TanhTableValueSize<5, 123, true>
    {
        typedef TanhValuesTableQ5_123 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_5_123

    #if TINYMIND_USE_TANH_6_122
    template<>
    struct TanhTableValueSize<6, 122, true>
    {
        typedef TanhValuesTableQ6_122 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_6_122

    #if TINYMIND_USE_TANH_7_121
    template<>
    struct TanhTableValueSize<7, 121, true>
    {
        typedef TanhValuesTableQ7_121 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_7_121

    #if TINYMIND_USE_TANH_8_120
    template<>
    struct TanhTableValueSize<8, 120, true>
    {
        typedef TanhValuesTableQ8_120 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_8_120

    #if TINYMIND_USE_TANH_9_119
    template<>
    struct TanhTableValueSize<9, 119, true>
    {
        typedef TanhValuesTableQ9_119 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_9_119

    #if TINYMIND_USE_TANH_10_118
    template<>
    struct TanhTableValueSize<10, 118, true>
    {
        typedef TanhValuesTableQ10_118 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_10_118

    #if TINYMIND_USE_TANH_11_117
    template<>
    struct TanhTableValueSize<11, 117, true>
    {
        typedef TanhValuesTableQ11_117 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_11_117

    #if TINYMIND_USE_TANH_12_116
    template<>
    struct TanhTableValueSize<12, 116, true>
    {
        typedef TanhValuesTableQ12_116 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_12_116

    #if TINYMIND_USE_TANH_13_115
    template<>
    struct TanhTableValueSize<13, 115, true>
    {
        typedef TanhValuesTableQ13_115 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_13_115

    #if TINYMIND_USE_TANH_14_114
    template<>
    struct TanhTableValueSize<14, 114, true>
    {
        typedef TanhValuesTableQ14_114 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_14_114

    #if TINYMIND_USE_TANH_15_113
    template<>
    struct TanhTableValueSize<15, 113, true>
    {
        typedef TanhValuesTableQ15_113 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_15_113

    #if TINYMIND_USE_TANH_16_112
    template<>
    struct TanhTableValueSize<16, 112, true>
    {
        typedef TanhValuesTableQ16_112 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_16_112

    #if TINYMIND_USE_TANH_17_111
    template<>
    struct TanhTableValueSize<17, 111, true>
    {
        typedef TanhValuesTableQ17_111 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_17_111

    #if TINYMIND_USE_TANH_18_110
    template<>
    struct TanhTableValueSize<18, 110, true>
    {
        typedef TanhValuesTableQ18_110 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_18_110

    #if TINYMIND_USE_TANH_19_109
    template<>
    struct TanhTableValueSize<19, 109, true>
    {
        typedef TanhValuesTableQ19_109 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_19_109

    #if TINYMIND_USE_TANH_20_108
    template<>
    struct TanhTableValueSize<20, 108, true>
    {
        typedef TanhValuesTableQ20_108 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_20_108

    #if TINYMIND_USE_TANH_21_107
    template<>
    struct TanhTableValueSize<21, 107, true>
    {
        typedef TanhValuesTableQ21_107 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_21_107

    #if TINYMIND_USE_TANH_22_106
    template<>
    struct TanhTableValueSize<22, 106, true>
    {
        typedef TanhValuesTableQ22_106 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_22_106

    #if TINYMIND_USE_TANH_23_105
    template<>
    struct TanhTableValueSize<23, 105, true>
    {
        typedef TanhValuesTableQ23_105 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_23_105

    #if TINYMIND_USE_TANH_24_104
    template<>
    struct TanhTableValueSize<24, 104, true>
    {
        typedef TanhValuesTableQ24_104 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_24_104

    #if TINYMIND_USE_TANH_25_103
    template<>
    struct TanhTableValueSize<25, 103, true>
    {
        typedef TanhValuesTableQ25_103 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_25_103

    #if TINYMIND_USE_TANH_26_102
    template<>
    struct TanhTableValueSize<26, 102, true>
    {
        typedef TanhValuesTableQ26_102 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_26_102

    #if TINYMIND_USE_TANH_27_101
    template<>
    struct TanhTableValueSize<27, 101, true>
    {
        typedef TanhValuesTableQ27_101 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_27_101

    #if TINYMIND_USE_TANH_28_100
    template<>
    struct TanhTableValueSize<28, 100, true>
    {
        typedef TanhValuesTableQ28_100 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_28_100

    #if TINYMIND_USE_TANH_29_99
    template<>
    struct TanhTableValueSize<29, 99, true>
    {
        typedef TanhValuesTableQ29_99 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_29_99

    #if TINYMIND_USE_TANH_30_98
    template<>
    struct TanhTableValueSize<30, 98, true>
    {
        typedef TanhValuesTableQ30_98 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_30_98

    #if TINYMIND_USE_TANH_31_97
    template<>
    struct TanhTableValueSize<31, 97, true>
    {
        typedef TanhValuesTableQ31_97 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_31_97

    #if TINYMIND_USE_TANH_32_96
    template<>
    struct TanhTableValueSize<32, 96, true>
    {
        typedef TanhValuesTableQ32_96 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_32_96

    #if TINYMIND_USE_TANH_33_95
    template<>
    struct TanhTableValueSize<33, 95, true>
    {
        typedef TanhValuesTableQ33_95 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_33_95

    #if TINYMIND_USE_TANH_34_94
    template<>
    struct TanhTableValueSize<34, 94, true>
    {
        typedef TanhValuesTableQ34_94 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_34_94

    #if TINYMIND_USE_TANH_35_93
    template<>
    struct TanhTableValueSize<35, 93, true>
    {
        typedef TanhValuesTableQ35_93 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_35_93

    #if TINYMIND_USE_TANH_36_92
    template<>
    struct TanhTableValueSize<36, 92, true>
    {
        typedef TanhValuesTableQ36_92 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_36_92

    #if TINYMIND_USE_TANH_37_91
    template<>
    struct TanhTableValueSize<37, 91, true>
    {
        typedef TanhValuesTableQ37_91 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_37_91

    #if TINYMIND_USE_TANH_38_90
    template<>
    struct TanhTableValueSize<38, 90, true>
    {
        typedef TanhValuesTableQ38_90 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_38_90

    #if TINYMIND_USE_TANH_39_89
    template<>
    struct TanhTableValueSize<39, 89, true>
    {
        typedef TanhValuesTableQ39_89 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_39_89

    #if TINYMIND_USE_TANH_40_88
    template<>
    struct TanhTableValueSize<40, 88, true>
    {
        typedef TanhValuesTableQ40_88 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_40_88

    #if TINYMIND_USE_TANH_41_87
    template<>
    struct TanhTableValueSize<41, 87, true>
    {
        typedef TanhValuesTableQ41_87 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_41_87

    #if TINYMIND_USE_TANH_42_86
    template<>
    struct TanhTableValueSize<42, 86, true>
    {
        typedef TanhValuesTableQ42_86 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_42_86

    #if TINYMIND_USE_TANH_43_85
    template<>
    struct TanhTableValueSize<43, 85, true>
    {
        typedef TanhValuesTableQ43_85 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_43_85

    #if TINYMIND_USE_TANH_44_84
    template<>
    struct TanhTableValueSize<44, 84, true>
    {
        typedef TanhValuesTableQ44_84 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_44_84

    #if TINYMIND_USE_TANH_45_83
    template<>
    struct TanhTableValueSize<45, 83, true>
    {
        typedef TanhValuesTableQ45_83 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_45_83

    #if TINYMIND_USE_TANH_46_82
    template<>
    struct TanhTableValueSize<46, 82, true>
    {
        typedef TanhValuesTableQ46_82 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_46_82

    #if TINYMIND_USE_TANH_47_81
    template<>
    struct TanhTableValueSize<47, 81, true>
    {
        typedef TanhValuesTableQ47_81 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_47_81

    #if TINYMIND_USE_TANH_48_80
    template<>
    struct TanhTableValueSize<48, 80, true>
    {
        typedef TanhValuesTableQ48_80 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_48_80

    #if TINYMIND_USE_TANH_49_79
    template<>
    struct TanhTableValueSize<49, 79, true>
    {
        typedef TanhValuesTableQ49_79 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_49_79

    #if TINYMIND_USE_TANH_50_78
    template<>
    struct TanhTableValueSize<50, 78, true>
    {
        typedef TanhValuesTableQ50_78 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_50_78

    #if TINYMIND_USE_TANH_51_77
    template<>
    struct TanhTableValueSize<51, 77, true>
    {
        typedef TanhValuesTableQ51_77 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_51_77

    #if TINYMIND_USE_TANH_52_76
    template<>
    struct TanhTableValueSize<52, 76, true>
    {
        typedef TanhValuesTableQ52_76 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_52_76

    #if TINYMIND_USE_TANH_53_75
    template<>
    struct TanhTableValueSize<53, 75, true>
    {
        typedef TanhValuesTableQ53_75 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_53_75

    #if TINYMIND_USE_TANH_54_74
    template<>
    struct TanhTableValueSize<54, 74, true>
    {
        typedef TanhValuesTableQ54_74 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_54_74

    #if TINYMIND_USE_TANH_55_73
    template<>
    struct TanhTableValueSize<55, 73, true>
    {
        typedef TanhValuesTableQ55_73 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_55_73

    #if TINYMIND_USE_TANH_56_72
    template<>
    struct TanhTableValueSize<56, 72, true>
    {
        typedef TanhValuesTableQ56_72 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_56_72

    #if TINYMIND_USE_TANH_57_71
    template<>
    struct TanhTableValueSize<57, 71, true>
    {
        typedef TanhValuesTableQ57_71 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_57_71

    #if TINYMIND_USE_TANH_58_70
    template<>
    struct TanhTableValueSize<58, 70, true>
    {
        typedef TanhValuesTableQ58_70 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_58_70

    #if TINYMIND_USE_TANH_59_69
    template<>
    struct TanhTableValueSize<59, 69, true>
    {
        typedef TanhValuesTableQ59_69 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_59_69

    #if TINYMIND_USE_TANH_60_68
    template<>
    struct TanhTableValueSize<60, 68, true>
    {
        typedef TanhValuesTableQ60_68 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_60_68

    #if TINYMIND_USE_TANH_61_67
    template<>
    struct TanhTableValueSize<61, 67, true>
    {
        typedef TanhValuesTableQ61_67 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_61_67

    #if TINYMIND_USE_TANH_62_66
    template<>
    struct TanhTableValueSize<62, 66, true>
    {
        typedef TanhValuesTableQ62_66 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_62_66

    #if TINYMIND_USE_TANH_63_65
    template<>
    struct TanhTableValueSize<63, 65, true>
    {
        typedef TanhValuesTableQ63_65 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_63_65

    #if TINYMIND_USE_TANH_64_64
    template<>
    struct TanhTableValueSize<64, 64, true>
    {
        typedef TanhValuesTableQ64_64 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_64_64

    #if TINYMIND_USE_TANH_65_63
    template<>
    struct TanhTableValueSize<65, 63, true>
    {
        typedef TanhValuesTableQ65_63 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_65_63

    #if TINYMIND_USE_TANH_66_62
    template<>
    struct TanhTableValueSize<66, 62, true>
    {
        typedef TanhValuesTableQ66_62 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_66_62

    #if TINYMIND_USE_TANH_67_61
    template<>
    struct TanhTableValueSize<67, 61, true>
    {
        typedef TanhValuesTableQ67_61 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_67_61

    #if TINYMIND_USE_TANH_68_60
    template<>
    struct TanhTableValueSize<68, 60, true>
    {
        typedef TanhValuesTableQ68_60 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_68_60

    #if TINYMIND_USE_TANH_69_59
    template<>
    struct TanhTableValueSize<69, 59, true>
    {
        typedef TanhValuesTableQ69_59 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_69_59

    #if TINYMIND_USE_TANH_70_58
    template<>
    struct TanhTableValueSize<70, 58, true>
    {
        typedef TanhValuesTableQ70_58 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_70_58

    #if TINYMIND_USE_TANH_71_57
    template<>
    struct TanhTableValueSize<71, 57, true>
    {
        typedef TanhValuesTableQ71_57 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_71_57

    #if TINYMIND_USE_TANH_72_56
    template<>
    struct TanhTableValueSize<72, 56, true>
    {
        typedef TanhValuesTableQ72_56 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_72_56

    #if TINYMIND_USE_TANH_73_55
    template<>
    struct TanhTableValueSize<73, 55, true>
    {
        typedef TanhValuesTableQ73_55 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_73_55

    #if TINYMIND_USE_TANH_74_54
    template<>
    struct TanhTableValueSize<74, 54, true>
    {
        typedef TanhValuesTableQ74_54 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_74_54

    #if TINYMIND_USE_TANH_75_53
    template<>
    struct TanhTableValueSize<75, 53, true>
    {
        typedef TanhValuesTableQ75_53 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_75_53

    #if TINYMIND_USE_TANH_76_52
    template<>
    struct TanhTableValueSize<76, 52, true>
    {
        typedef TanhValuesTableQ76_52 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_76_52

    #if TINYMIND_USE_TANH_77_51
    template<>
    struct TanhTableValueSize<77, 51, true>
    {
        typedef TanhValuesTableQ77_51 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_77_51

    #if TINYMIND_USE_TANH_78_50
    template<>
    struct TanhTableValueSize<78, 50, true>
    {
        typedef TanhValuesTableQ78_50 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_78_50

    #if TINYMIND_USE_TANH_79_49
    template<>
    struct TanhTableValueSize<79, 49, true>
    {
        typedef TanhValuesTableQ79_49 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_79_49

    #if TINYMIND_USE_TANH_80_48
    template<>
    struct TanhTableValueSize<80, 48, true>
    {
        typedef TanhValuesTableQ80_48 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_80_48

    #if TINYMIND_USE_TANH_81_47
    template<>
    struct TanhTableValueSize<81, 47, true>
    {
        typedef TanhValuesTableQ81_47 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_81_47

    #if TINYMIND_USE_TANH_82_46
    template<>
    struct TanhTableValueSize<82, 46, true>
    {
        typedef TanhValuesTableQ82_46 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_82_46

    #if TINYMIND_USE_TANH_83_45
    template<>
    struct TanhTableValueSize<83, 45, true>
    {
        typedef TanhValuesTableQ83_45 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_83_45

    #if TINYMIND_USE_TANH_84_44
    template<>
    struct TanhTableValueSize<84, 44, true>
    {
        typedef TanhValuesTableQ84_44 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_84_44

    #if TINYMIND_USE_TANH_85_43
    template<>
    struct TanhTableValueSize<85, 43, true>
    {
        typedef TanhValuesTableQ85_43 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_85_43

    #if TINYMIND_USE_TANH_86_42
    template<>
    struct TanhTableValueSize<86, 42, true>
    {
        typedef TanhValuesTableQ86_42 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_86_42

    #if TINYMIND_USE_TANH_87_41
    template<>
    struct TanhTableValueSize<87, 41, true>
    {
        typedef TanhValuesTableQ87_41 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_87_41

    #if TINYMIND_USE_TANH_88_40
    template<>
    struct TanhTableValueSize<88, 40, true>
    {
        typedef TanhValuesTableQ88_40 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_88_40

    #if TINYMIND_USE_TANH_89_39
    template<>
    struct TanhTableValueSize<89, 39, true>
    {
        typedef TanhValuesTableQ89_39 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_89_39

    #if TINYMIND_USE_TANH_90_38
    template<>
    struct TanhTableValueSize<90, 38, true>
    {
        typedef TanhValuesTableQ90_38 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_90_38

    #if TINYMIND_USE_TANH_91_37
    template<>
    struct TanhTableValueSize<91, 37, true>
    {
        typedef TanhValuesTableQ91_37 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_91_37

    #if TINYMIND_USE_TANH_92_36
    template<>
    struct TanhTableValueSize<92, 36, true>
    {
        typedef TanhValuesTableQ92_36 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_92_36

    #if TINYMIND_USE_TANH_93_35
    template<>
    struct TanhTableValueSize<93, 35, true>
    {
        typedef TanhValuesTableQ93_35 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_93_35

    #if TINYMIND_USE_TANH_94_34
    template<>
    struct TanhTableValueSize<94, 34, true>
    {
        typedef TanhValuesTableQ94_34 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_94_34

    #if TINYMIND_USE_TANH_95_33
    template<>
    struct TanhTableValueSize<95, 33, true>
    {
        typedef TanhValuesTableQ95_33 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_95_33

    #if TINYMIND_USE_TANH_96_32
    template<>
    struct TanhTableValueSize<96, 32, true>
    {
        typedef TanhValuesTableQ96_32 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_96_32

    #if TINYMIND_USE_TANH_97_31
    template<>
    struct TanhTableValueSize<97, 31, true>
    {
        typedef TanhValuesTableQ97_31 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_97_31

    #if TINYMIND_USE_TANH_98_30
    template<>
    struct TanhTableValueSize<98, 30, true>
    {
        typedef TanhValuesTableQ98_30 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_98_30

    #if TINYMIND_USE_TANH_99_29
    template<>
    struct TanhTableValueSize<99, 29, true>
    {
        typedef TanhValuesTableQ99_29 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_99_29

    #if TINYMIND_USE_TANH_100_28
    template<>
    struct TanhTableValueSize<100, 28, true>
    {
        typedef TanhValuesTableQ100_28 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_100_28

    #if TINYMIND_USE_TANH_101_27
    template<>
    struct TanhTableValueSize<101, 27, true>
    {
        typedef TanhValuesTableQ101_27 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_101_27

    #if TINYMIND_USE_TANH_102_26
    template<>
    struct TanhTableValueSize<102, 26, true>
    {
        typedef TanhValuesTableQ102_26 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_102_26

    #if TINYMIND_USE_TANH_103_25
    template<>
    struct TanhTableValueSize<103, 25, true>
    {
        typedef TanhValuesTableQ103_25 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_103_25

    #if TINYMIND_USE_TANH_104_24
    template<>
    struct TanhTableValueSize<104, 24, true>
    {
        typedef TanhValuesTableQ104_24 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_104_24

    #if TINYMIND_USE_TANH_105_23
    template<>
    struct TanhTableValueSize<105, 23, true>
    {
        typedef TanhValuesTableQ105_23 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_105_23

    #if TINYMIND_USE_TANH_106_22
    template<>
    struct TanhTableValueSize<106, 22, true>
    {
        typedef TanhValuesTableQ106_22 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_106_22

    #if TINYMIND_USE_TANH_107_21
    template<>
    struct TanhTableValueSize<107, 21, true>
    {
        typedef TanhValuesTableQ107_21 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_107_21

    #if TINYMIND_USE_TANH_108_20
    template<>
    struct TanhTableValueSize<108, 20, true>
    {
        typedef TanhValuesTableQ108_20 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_108_20

    #if TINYMIND_USE_TANH_109_19
    template<>
    struct TanhTableValueSize<109, 19, true>
    {
        typedef TanhValuesTableQ109_19 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_109_19

    #if TINYMIND_USE_TANH_110_18
    template<>
    struct TanhTableValueSize<110, 18, true>
    {
        typedef TanhValuesTableQ110_18 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_110_18

    #if TINYMIND_USE_TANH_111_17
    template<>
    struct TanhTableValueSize<111, 17, true>
    {
        typedef TanhValuesTableQ111_17 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_111_17

    #if TINYMIND_USE_TANH_112_16
    template<>
    struct TanhTableValueSize<112, 16, true>
    {
        typedef TanhValuesTableQ112_16 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_112_16

    #if TINYMIND_USE_TANH_113_15
    template<>
    struct TanhTableValueSize<113, 15, true>
    {
        typedef TanhValuesTableQ113_15 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_113_15

    #if TINYMIND_USE_TANH_114_14
    template<>
    struct TanhTableValueSize<114, 14, true>
    {
        typedef TanhValuesTableQ114_14 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_114_14

    #if TINYMIND_USE_TANH_115_13
    template<>
    struct TanhTableValueSize<115, 13, true>
    {
        typedef TanhValuesTableQ115_13 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_115_13

    #if TINYMIND_USE_TANH_116_12
    template<>
    struct TanhTableValueSize<116, 12, true>
    {
        typedef TanhValuesTableQ116_12 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_116_12

    #if TINYMIND_USE_TANH_117_11
    template<>
    struct TanhTableValueSize<117, 11, true>
    {
        typedef TanhValuesTableQ117_11 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_117_11

    #if TINYMIND_USE_TANH_118_10
    template<>
    struct TanhTableValueSize<118, 10, true>
    {
        typedef TanhValuesTableQ118_10 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_118_10

    #if TINYMIND_USE_TANH_119_9
    template<>
    struct TanhTableValueSize<119, 9, true>
    {
        typedef TanhValuesTableQ119_9 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_119_9

    #if TINYMIND_USE_TANH_120_8
    template<>
    struct TanhTableValueSize<120, 8, true>
    {
        typedef TanhValuesTableQ120_8 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_120_8

    #if TINYMIND_USE_TANH_121_7
    template<>
    struct TanhTableValueSize<121, 7, true>
    {
        typedef TanhValuesTableQ121_7 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_121_7

    #if TINYMIND_USE_TANH_122_6
    template<>
    struct TanhTableValueSize<122, 6, true>
    {
        typedef TanhValuesTableQ122_6 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_122_6

    #if TINYMIND_USE_TANH_123_5
    template<>
    struct TanhTableValueSize<123, 5, true>
    {
        typedef TanhValuesTableQ123_5 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_123_5

    #if TINYMIND_USE_TANH_124_4
    template<>
    struct TanhTableValueSize<124, 4, true>
    {
        typedef TanhValuesTableQ124_4 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_124_4

    #if TINYMIND_USE_TANH_125_3
    template<>
    struct TanhTableValueSize<125, 3, true>
    {
        typedef TanhValuesTableQ125_3 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_125_3

    #if TINYMIND_USE_TANH_126_2
    template<>
    struct TanhTableValueSize<126, 2, true>
    {
        typedef TanhValuesTableQ126_2 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_126_2

    #if TINYMIND_USE_TANH_127_1
    template<>
    struct TanhTableValueSize<127, 1, true>
    {
        typedef TanhValuesTableQ127_1 TanhTableType;
    };
    #endif // TINYMIND_USE_TANH_127_1

    template<unsigned FixedBits,unsigned FracBits, bool IsSigned>
    struct TanhValuesTableSelector
    {
        typedef typename TanhTableValueSize<FixedBits, FracBits, IsSigned>::TanhTableType TanhTableType;
    };
}
