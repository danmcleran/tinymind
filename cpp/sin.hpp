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

#pragma once

#include "activation.hpp"

#include "sinValues8Bit.hpp"
#include "sinValues16Bit.hpp"
#include "sinValues32Bit.hpp"
#include "sinValues64Bit.hpp"
#include "sinValues128Bit.hpp"

namespace tinymind {
    template<unsigned FixedBits, unsigned FracBits, bool IsSigned>
    struct SinTableValueSize
    {
    };

    #if TINYMIND_USE_SIN_1_7
    template<>
    struct SinTableValueSize<1, 7, true>
    {
        typedef SinValuesTableQ1_7 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_1_7

    #if TINYMIND_USE_SIN_2_6
    template<>
    struct SinTableValueSize<2, 6, true>
    {
        typedef SinValuesTableQ2_6 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_2_6

    #if TINYMIND_USE_SIN_3_5
    template<>
    struct SinTableValueSize<3, 5, true>
    {
        typedef SinValuesTableQ3_5 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_3_5

    #if TINYMIND_USE_SIN_4_4
    template<>
    struct SinTableValueSize<4, 4, true>
    {
        typedef SinValuesTableQ4_4 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_4_4

    #if TINYMIND_USE_SIN_5_3
    template<>
    struct SinTableValueSize<5, 3, true>
    {
        typedef SinValuesTableQ5_3 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_5_3

    #if TINYMIND_USE_SIN_6_2
    template<>
    struct SinTableValueSize<6, 2, true>
    {
        typedef SinValuesTableQ6_2 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_6_2

    #if TINYMIND_USE_SIN_7_1
    template<>
    struct SinTableValueSize<7, 1, true>
    {
        typedef SinValuesTableQ7_1 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_7_1

    #if TINYMIND_USE_SIN_1_15
    template<>
    struct SinTableValueSize<1, 15, true>
    {
        typedef SinValuesTableQ1_15 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_1_15

    #if TINYMIND_USE_SIN_2_14
    template<>
    struct SinTableValueSize<2, 14, true>
    {
        typedef SinValuesTableQ2_14 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_2_14

    #if TINYMIND_USE_SIN_3_13
    template<>
    struct SinTableValueSize<3, 13, true>
    {
        typedef SinValuesTableQ3_13 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_3_13

    #if TINYMIND_USE_SIN_4_12
    template<>
    struct SinTableValueSize<4, 12, true>
    {
        typedef SinValuesTableQ4_12 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_4_12

    #if TINYMIND_USE_SIN_5_11
    template<>
    struct SinTableValueSize<5, 11, true>
    {
        typedef SinValuesTableQ5_11 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_5_11

    #if TINYMIND_USE_SIN_6_10
    template<>
    struct SinTableValueSize<6, 10, true>
    {
        typedef SinValuesTableQ6_10 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_6_10

    #if TINYMIND_USE_SIN_7_9
    template<>
    struct SinTableValueSize<7, 9, true>
    {
        typedef SinValuesTableQ7_9 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_7_9

    #if TINYMIND_USE_SIN_8_8
    template<>
    struct SinTableValueSize<8, 8, true>
    {
        typedef SinValuesTableQ8_8 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_8_8

    #if TINYMIND_USE_SIN_9_7
    template<>
    struct SinTableValueSize<9, 7, true>
    {
        typedef SinValuesTableQ9_7 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_9_7

    #if TINYMIND_USE_SIN_10_6
    template<>
    struct SinTableValueSize<10, 6, true>
    {
        typedef SinValuesTableQ10_6 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_10_6

    #if TINYMIND_USE_SIN_11_5
    template<>
    struct SinTableValueSize<11, 5, true>
    {
        typedef SinValuesTableQ11_5 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_11_5

    #if TINYMIND_USE_SIN_12_4
    template<>
    struct SinTableValueSize<12, 4, true>
    {
        typedef SinValuesTableQ12_4 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_12_4

    #if TINYMIND_USE_SIN_13_3
    template<>
    struct SinTableValueSize<13, 3, true>
    {
        typedef SinValuesTableQ13_3 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_13_3

    #if TINYMIND_USE_SIN_14_2
    template<>
    struct SinTableValueSize<14, 2, true>
    {
        typedef SinValuesTableQ14_2 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_14_2

    #if TINYMIND_USE_SIN_15_1
    template<>
    struct SinTableValueSize<15, 1, true>
    {
        typedef SinValuesTableQ15_1 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_15_1

    #if TINYMIND_USE_SIN_1_31
    template<>
    struct SinTableValueSize<1, 31, true>
    {
        typedef SinValuesTableQ1_31 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_1_31

    #if TINYMIND_USE_SIN_2_30
    template<>
    struct SinTableValueSize<2, 30, true>
    {
        typedef SinValuesTableQ2_30 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_2_30

    #if TINYMIND_USE_SIN_3_29
    template<>
    struct SinTableValueSize<3, 29, true>
    {
        typedef SinValuesTableQ3_29 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_3_29

    #if TINYMIND_USE_SIN_4_28
    template<>
    struct SinTableValueSize<4, 28, true>
    {
        typedef SinValuesTableQ4_28 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_4_28

    #if TINYMIND_USE_SIN_5_27
    template<>
    struct SinTableValueSize<5, 27, true>
    {
        typedef SinValuesTableQ5_27 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_5_27

    #if TINYMIND_USE_SIN_6_26
    template<>
    struct SinTableValueSize<6, 26, true>
    {
        typedef SinValuesTableQ6_26 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_6_26

    #if TINYMIND_USE_SIN_7_25
    template<>
    struct SinTableValueSize<7, 25, true>
    {
        typedef SinValuesTableQ7_25 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_7_25

    #if TINYMIND_USE_SIN_8_24
    template<>
    struct SinTableValueSize<8, 24, true>
    {
        typedef SinValuesTableQ8_24 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_8_24

    #if TINYMIND_USE_SIN_9_23
    template<>
    struct SinTableValueSize<9, 23, true>
    {
        typedef SinValuesTableQ9_23 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_9_23

    #if TINYMIND_USE_SIN_10_22
    template<>
    struct SinTableValueSize<10, 22, true>
    {
        typedef SinValuesTableQ10_22 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_10_22

    #if TINYMIND_USE_SIN_11_21
    template<>
    struct SinTableValueSize<11, 21, true>
    {
        typedef SinValuesTableQ11_21 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_11_21

    #if TINYMIND_USE_SIN_12_20
    template<>
    struct SinTableValueSize<12, 20, true>
    {
        typedef SinValuesTableQ12_20 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_12_20

    #if TINYMIND_USE_SIN_13_19
    template<>
    struct SinTableValueSize<13, 19, true>
    {
        typedef SinValuesTableQ13_19 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_13_19

    #if TINYMIND_USE_SIN_14_18
    template<>
    struct SinTableValueSize<14, 18, true>
    {
        typedef SinValuesTableQ14_18 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_14_18

    #if TINYMIND_USE_SIN_15_17
    template<>
    struct SinTableValueSize<15, 17, true>
    {
        typedef SinValuesTableQ15_17 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_15_17

    #if TINYMIND_USE_SIN_16_16
    template<>
    struct SinTableValueSize<16, 16, true>
    {
        typedef SinValuesTableQ16_16 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_16_16

    #if TINYMIND_USE_SIN_17_15
    template<>
    struct SinTableValueSize<17, 15, true>
    {
        typedef SinValuesTableQ17_15 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_17_15

    #if TINYMIND_USE_SIN_18_14
    template<>
    struct SinTableValueSize<18, 14, true>
    {
        typedef SinValuesTableQ18_14 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_18_14

    #if TINYMIND_USE_SIN_19_13
    template<>
    struct SinTableValueSize<19, 13, true>
    {
        typedef SinValuesTableQ19_13 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_19_13

    #if TINYMIND_USE_SIN_20_12
    template<>
    struct SinTableValueSize<20, 12, true>
    {
        typedef SinValuesTableQ20_12 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_20_12

    #if TINYMIND_USE_SIN_21_11
    template<>
    struct SinTableValueSize<21, 11, true>
    {
        typedef SinValuesTableQ21_11 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_21_11

    #if TINYMIND_USE_SIN_22_10
    template<>
    struct SinTableValueSize<22, 10, true>
    {
        typedef SinValuesTableQ22_10 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_22_10

    #if TINYMIND_USE_SIN_23_9
    template<>
    struct SinTableValueSize<23, 9, true>
    {
        typedef SinValuesTableQ23_9 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_23_9

    #if TINYMIND_USE_SIN_24_8
    template<>
    struct SinTableValueSize<24, 8, true>
    {
        typedef SinValuesTableQ24_8 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_24_8

    #if TINYMIND_USE_SIN_25_7
    template<>
    struct SinTableValueSize<25, 7, true>
    {
        typedef SinValuesTableQ25_7 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_25_7

    #if TINYMIND_USE_SIN_26_6
    template<>
    struct SinTableValueSize<26, 6, true>
    {
        typedef SinValuesTableQ26_6 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_26_6

    #if TINYMIND_USE_SIN_27_5
    template<>
    struct SinTableValueSize<27, 5, true>
    {
        typedef SinValuesTableQ27_5 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_27_5

    #if TINYMIND_USE_SIN_28_4
    template<>
    struct SinTableValueSize<28, 4, true>
    {
        typedef SinValuesTableQ28_4 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_28_4

    #if TINYMIND_USE_SIN_29_3
    template<>
    struct SinTableValueSize<29, 3, true>
    {
        typedef SinValuesTableQ29_3 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_29_3

    #if TINYMIND_USE_SIN_30_2
    template<>
    struct SinTableValueSize<30, 2, true>
    {
        typedef SinValuesTableQ30_2 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_30_2

    #if TINYMIND_USE_SIN_31_1
    template<>
    struct SinTableValueSize<31, 1, true>
    {
        typedef SinValuesTableQ31_1 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_31_1

    #if TINYMIND_USE_SIN_1_63
    template<>
    struct SinTableValueSize<1, 63, true>
    {
        typedef SinValuesTableQ1_63 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_1_63

    #if TINYMIND_USE_SIN_2_62
    template<>
    struct SinTableValueSize<2, 62, true>
    {
        typedef SinValuesTableQ2_62 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_2_62

    #if TINYMIND_USE_SIN_3_61
    template<>
    struct SinTableValueSize<3, 61, true>
    {
        typedef SinValuesTableQ3_61 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_3_61

    #if TINYMIND_USE_SIN_4_60
    template<>
    struct SinTableValueSize<4, 60, true>
    {
        typedef SinValuesTableQ4_60 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_4_60

    #if TINYMIND_USE_SIN_5_59
    template<>
    struct SinTableValueSize<5, 59, true>
    {
        typedef SinValuesTableQ5_59 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_5_59

    #if TINYMIND_USE_SIN_6_58
    template<>
    struct SinTableValueSize<6, 58, true>
    {
        typedef SinValuesTableQ6_58 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_6_58

    #if TINYMIND_USE_SIN_7_57
    template<>
    struct SinTableValueSize<7, 57, true>
    {
        typedef SinValuesTableQ7_57 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_7_57

    #if TINYMIND_USE_SIN_8_56
    template<>
    struct SinTableValueSize<8, 56, true>
    {
        typedef SinValuesTableQ8_56 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_8_56

    #if TINYMIND_USE_SIN_9_55
    template<>
    struct SinTableValueSize<9, 55, true>
    {
        typedef SinValuesTableQ9_55 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_9_55

    #if TINYMIND_USE_SIN_10_54
    template<>
    struct SinTableValueSize<10, 54, true>
    {
        typedef SinValuesTableQ10_54 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_10_54

    #if TINYMIND_USE_SIN_11_53
    template<>
    struct SinTableValueSize<11, 53, true>
    {
        typedef SinValuesTableQ11_53 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_11_53

    #if TINYMIND_USE_SIN_12_52
    template<>
    struct SinTableValueSize<12, 52, true>
    {
        typedef SinValuesTableQ12_52 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_12_52

    #if TINYMIND_USE_SIN_13_51
    template<>
    struct SinTableValueSize<13, 51, true>
    {
        typedef SinValuesTableQ13_51 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_13_51

    #if TINYMIND_USE_SIN_14_50
    template<>
    struct SinTableValueSize<14, 50, true>
    {
        typedef SinValuesTableQ14_50 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_14_50

    #if TINYMIND_USE_SIN_15_49
    template<>
    struct SinTableValueSize<15, 49, true>
    {
        typedef SinValuesTableQ15_49 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_15_49

    #if TINYMIND_USE_SIN_16_48
    template<>
    struct SinTableValueSize<16, 48, true>
    {
        typedef SinValuesTableQ16_48 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_16_48

    #if TINYMIND_USE_SIN_17_47
    template<>
    struct SinTableValueSize<17, 47, true>
    {
        typedef SinValuesTableQ17_47 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_17_47

    #if TINYMIND_USE_SIN_18_46
    template<>
    struct SinTableValueSize<18, 46, true>
    {
        typedef SinValuesTableQ18_46 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_18_46

    #if TINYMIND_USE_SIN_19_45
    template<>
    struct SinTableValueSize<19, 45, true>
    {
        typedef SinValuesTableQ19_45 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_19_45

    #if TINYMIND_USE_SIN_20_44
    template<>
    struct SinTableValueSize<20, 44, true>
    {
        typedef SinValuesTableQ20_44 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_20_44

    #if TINYMIND_USE_SIN_21_43
    template<>
    struct SinTableValueSize<21, 43, true>
    {
        typedef SinValuesTableQ21_43 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_21_43

    #if TINYMIND_USE_SIN_22_42
    template<>
    struct SinTableValueSize<22, 42, true>
    {
        typedef SinValuesTableQ22_42 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_22_42

    #if TINYMIND_USE_SIN_23_41
    template<>
    struct SinTableValueSize<23, 41, true>
    {
        typedef SinValuesTableQ23_41 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_23_41

    #if TINYMIND_USE_SIN_24_40
    template<>
    struct SinTableValueSize<24, 40, true>
    {
        typedef SinValuesTableQ24_40 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_24_40

    #if TINYMIND_USE_SIN_25_39
    template<>
    struct SinTableValueSize<25, 39, true>
    {
        typedef SinValuesTableQ25_39 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_25_39

    #if TINYMIND_USE_SIN_26_38
    template<>
    struct SinTableValueSize<26, 38, true>
    {
        typedef SinValuesTableQ26_38 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_26_38

    #if TINYMIND_USE_SIN_27_37
    template<>
    struct SinTableValueSize<27, 37, true>
    {
        typedef SinValuesTableQ27_37 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_27_37

    #if TINYMIND_USE_SIN_28_36
    template<>
    struct SinTableValueSize<28, 36, true>
    {
        typedef SinValuesTableQ28_36 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_28_36

    #if TINYMIND_USE_SIN_29_35
    template<>
    struct SinTableValueSize<29, 35, true>
    {
        typedef SinValuesTableQ29_35 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_29_35

    #if TINYMIND_USE_SIN_30_34
    template<>
    struct SinTableValueSize<30, 34, true>
    {
        typedef SinValuesTableQ30_34 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_30_34

    #if TINYMIND_USE_SIN_31_33
    template<>
    struct SinTableValueSize<31, 33, true>
    {
        typedef SinValuesTableQ31_33 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_31_33

    #if TINYMIND_USE_SIN_32_32
    template<>
    struct SinTableValueSize<32, 32, true>
    {
        typedef SinValuesTableQ32_32 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_32_32

    #if TINYMIND_USE_SIN_33_31
    template<>
    struct SinTableValueSize<33, 31, true>
    {
        typedef SinValuesTableQ33_31 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_33_31

    #if TINYMIND_USE_SIN_34_30
    template<>
    struct SinTableValueSize<34, 30, true>
    {
        typedef SinValuesTableQ34_30 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_34_30

    #if TINYMIND_USE_SIN_35_29
    template<>
    struct SinTableValueSize<35, 29, true>
    {
        typedef SinValuesTableQ35_29 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_35_29

    #if TINYMIND_USE_SIN_36_28
    template<>
    struct SinTableValueSize<36, 28, true>
    {
        typedef SinValuesTableQ36_28 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_36_28

    #if TINYMIND_USE_SIN_37_27
    template<>
    struct SinTableValueSize<37, 27, true>
    {
        typedef SinValuesTableQ37_27 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_37_27

    #if TINYMIND_USE_SIN_38_26
    template<>
    struct SinTableValueSize<38, 26, true>
    {
        typedef SinValuesTableQ38_26 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_38_26

    #if TINYMIND_USE_SIN_39_25
    template<>
    struct SinTableValueSize<39, 25, true>
    {
        typedef SinValuesTableQ39_25 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_39_25

    #if TINYMIND_USE_SIN_40_24
    template<>
    struct SinTableValueSize<40, 24, true>
    {
        typedef SinValuesTableQ40_24 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_40_24

    #if TINYMIND_USE_SIN_41_23
    template<>
    struct SinTableValueSize<41, 23, true>
    {
        typedef SinValuesTableQ41_23 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_41_23

    #if TINYMIND_USE_SIN_42_22
    template<>
    struct SinTableValueSize<42, 22, true>
    {
        typedef SinValuesTableQ42_22 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_42_22

    #if TINYMIND_USE_SIN_43_21
    template<>
    struct SinTableValueSize<43, 21, true>
    {
        typedef SinValuesTableQ43_21 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_43_21

    #if TINYMIND_USE_SIN_44_20
    template<>
    struct SinTableValueSize<44, 20, true>
    {
        typedef SinValuesTableQ44_20 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_44_20

    #if TINYMIND_USE_SIN_45_19
    template<>
    struct SinTableValueSize<45, 19, true>
    {
        typedef SinValuesTableQ45_19 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_45_19

    #if TINYMIND_USE_SIN_46_18
    template<>
    struct SinTableValueSize<46, 18, true>
    {
        typedef SinValuesTableQ46_18 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_46_18

    #if TINYMIND_USE_SIN_47_17
    template<>
    struct SinTableValueSize<47, 17, true>
    {
        typedef SinValuesTableQ47_17 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_47_17

    #if TINYMIND_USE_SIN_48_16
    template<>
    struct SinTableValueSize<48, 16, true>
    {
        typedef SinValuesTableQ48_16 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_48_16

    #if TINYMIND_USE_SIN_49_15
    template<>
    struct SinTableValueSize<49, 15, true>
    {
        typedef SinValuesTableQ49_15 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_49_15

    #if TINYMIND_USE_SIN_50_14
    template<>
    struct SinTableValueSize<50, 14, true>
    {
        typedef SinValuesTableQ50_14 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_50_14

    #if TINYMIND_USE_SIN_51_13
    template<>
    struct SinTableValueSize<51, 13, true>
    {
        typedef SinValuesTableQ51_13 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_51_13

    #if TINYMIND_USE_SIN_52_12
    template<>
    struct SinTableValueSize<52, 12, true>
    {
        typedef SinValuesTableQ52_12 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_52_12

    #if TINYMIND_USE_SIN_53_11
    template<>
    struct SinTableValueSize<53, 11, true>
    {
        typedef SinValuesTableQ53_11 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_53_11

    #if TINYMIND_USE_SIN_54_10
    template<>
    struct SinTableValueSize<54, 10, true>
    {
        typedef SinValuesTableQ54_10 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_54_10

    #if TINYMIND_USE_SIN_55_9
    template<>
    struct SinTableValueSize<55, 9, true>
    {
        typedef SinValuesTableQ55_9 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_55_9

    #if TINYMIND_USE_SIN_56_8
    template<>
    struct SinTableValueSize<56, 8, true>
    {
        typedef SinValuesTableQ56_8 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_56_8

    #if TINYMIND_USE_SIN_57_7
    template<>
    struct SinTableValueSize<57, 7, true>
    {
        typedef SinValuesTableQ57_7 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_57_7

    #if TINYMIND_USE_SIN_58_6
    template<>
    struct SinTableValueSize<58, 6, true>
    {
        typedef SinValuesTableQ58_6 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_58_6

    #if TINYMIND_USE_SIN_59_5
    template<>
    struct SinTableValueSize<59, 5, true>
    {
        typedef SinValuesTableQ59_5 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_59_5

    #if TINYMIND_USE_SIN_60_4
    template<>
    struct SinTableValueSize<60, 4, true>
    {
        typedef SinValuesTableQ60_4 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_60_4

    #if TINYMIND_USE_SIN_61_3
    template<>
    struct SinTableValueSize<61, 3, true>
    {
        typedef SinValuesTableQ61_3 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_61_3

    #if TINYMIND_USE_SIN_62_2
    template<>
    struct SinTableValueSize<62, 2, true>
    {
        typedef SinValuesTableQ62_2 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_62_2

    #if TINYMIND_USE_SIN_63_1
    template<>
    struct SinTableValueSize<63, 1, true>
    {
        typedef SinValuesTableQ63_1 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_63_1

    #if TINYMIND_USE_SIN_1_127
    template<>
    struct SinTableValueSize<1, 127, true>
    {
        typedef SinValuesTableQ1_127 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_1_127

    #if TINYMIND_USE_SIN_2_126
    template<>
    struct SinTableValueSize<2, 126, true>
    {
        typedef SinValuesTableQ2_126 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_2_126

    #if TINYMIND_USE_SIN_3_125
    template<>
    struct SinTableValueSize<3, 125, true>
    {
        typedef SinValuesTableQ3_125 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_3_125

    #if TINYMIND_USE_SIN_4_124
    template<>
    struct SinTableValueSize<4, 124, true>
    {
        typedef SinValuesTableQ4_124 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_4_124

    #if TINYMIND_USE_SIN_5_123
    template<>
    struct SinTableValueSize<5, 123, true>
    {
        typedef SinValuesTableQ5_123 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_5_123

    #if TINYMIND_USE_SIN_6_122
    template<>
    struct SinTableValueSize<6, 122, true>
    {
        typedef SinValuesTableQ6_122 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_6_122

    #if TINYMIND_USE_SIN_7_121
    template<>
    struct SinTableValueSize<7, 121, true>
    {
        typedef SinValuesTableQ7_121 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_7_121

    #if TINYMIND_USE_SIN_8_120
    template<>
    struct SinTableValueSize<8, 120, true>
    {
        typedef SinValuesTableQ8_120 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_8_120

    #if TINYMIND_USE_SIN_9_119
    template<>
    struct SinTableValueSize<9, 119, true>
    {
        typedef SinValuesTableQ9_119 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_9_119

    #if TINYMIND_USE_SIN_10_118
    template<>
    struct SinTableValueSize<10, 118, true>
    {
        typedef SinValuesTableQ10_118 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_10_118

    #if TINYMIND_USE_SIN_11_117
    template<>
    struct SinTableValueSize<11, 117, true>
    {
        typedef SinValuesTableQ11_117 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_11_117

    #if TINYMIND_USE_SIN_12_116
    template<>
    struct SinTableValueSize<12, 116, true>
    {
        typedef SinValuesTableQ12_116 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_12_116

    #if TINYMIND_USE_SIN_13_115
    template<>
    struct SinTableValueSize<13, 115, true>
    {
        typedef SinValuesTableQ13_115 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_13_115

    #if TINYMIND_USE_SIN_14_114
    template<>
    struct SinTableValueSize<14, 114, true>
    {
        typedef SinValuesTableQ14_114 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_14_114

    #if TINYMIND_USE_SIN_15_113
    template<>
    struct SinTableValueSize<15, 113, true>
    {
        typedef SinValuesTableQ15_113 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_15_113

    #if TINYMIND_USE_SIN_16_112
    template<>
    struct SinTableValueSize<16, 112, true>
    {
        typedef SinValuesTableQ16_112 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_16_112

    #if TINYMIND_USE_SIN_17_111
    template<>
    struct SinTableValueSize<17, 111, true>
    {
        typedef SinValuesTableQ17_111 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_17_111

    #if TINYMIND_USE_SIN_18_110
    template<>
    struct SinTableValueSize<18, 110, true>
    {
        typedef SinValuesTableQ18_110 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_18_110

    #if TINYMIND_USE_SIN_19_109
    template<>
    struct SinTableValueSize<19, 109, true>
    {
        typedef SinValuesTableQ19_109 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_19_109

    #if TINYMIND_USE_SIN_20_108
    template<>
    struct SinTableValueSize<20, 108, true>
    {
        typedef SinValuesTableQ20_108 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_20_108

    #if TINYMIND_USE_SIN_21_107
    template<>
    struct SinTableValueSize<21, 107, true>
    {
        typedef SinValuesTableQ21_107 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_21_107

    #if TINYMIND_USE_SIN_22_106
    template<>
    struct SinTableValueSize<22, 106, true>
    {
        typedef SinValuesTableQ22_106 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_22_106

    #if TINYMIND_USE_SIN_23_105
    template<>
    struct SinTableValueSize<23, 105, true>
    {
        typedef SinValuesTableQ23_105 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_23_105

    #if TINYMIND_USE_SIN_24_104
    template<>
    struct SinTableValueSize<24, 104, true>
    {
        typedef SinValuesTableQ24_104 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_24_104

    #if TINYMIND_USE_SIN_25_103
    template<>
    struct SinTableValueSize<25, 103, true>
    {
        typedef SinValuesTableQ25_103 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_25_103

    #if TINYMIND_USE_SIN_26_102
    template<>
    struct SinTableValueSize<26, 102, true>
    {
        typedef SinValuesTableQ26_102 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_26_102

    #if TINYMIND_USE_SIN_27_101
    template<>
    struct SinTableValueSize<27, 101, true>
    {
        typedef SinValuesTableQ27_101 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_27_101

    #if TINYMIND_USE_SIN_28_100
    template<>
    struct SinTableValueSize<28, 100, true>
    {
        typedef SinValuesTableQ28_100 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_28_100

    #if TINYMIND_USE_SIN_29_99
    template<>
    struct SinTableValueSize<29, 99, true>
    {
        typedef SinValuesTableQ29_99 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_29_99

    #if TINYMIND_USE_SIN_30_98
    template<>
    struct SinTableValueSize<30, 98, true>
    {
        typedef SinValuesTableQ30_98 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_30_98

    #if TINYMIND_USE_SIN_31_97
    template<>
    struct SinTableValueSize<31, 97, true>
    {
        typedef SinValuesTableQ31_97 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_31_97

    #if TINYMIND_USE_SIN_32_96
    template<>
    struct SinTableValueSize<32, 96, true>
    {
        typedef SinValuesTableQ32_96 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_32_96

    #if TINYMIND_USE_SIN_33_95
    template<>
    struct SinTableValueSize<33, 95, true>
    {
        typedef SinValuesTableQ33_95 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_33_95

    #if TINYMIND_USE_SIN_34_94
    template<>
    struct SinTableValueSize<34, 94, true>
    {
        typedef SinValuesTableQ34_94 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_34_94

    #if TINYMIND_USE_SIN_35_93
    template<>
    struct SinTableValueSize<35, 93, true>
    {
        typedef SinValuesTableQ35_93 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_35_93

    #if TINYMIND_USE_SIN_36_92
    template<>
    struct SinTableValueSize<36, 92, true>
    {
        typedef SinValuesTableQ36_92 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_36_92

    #if TINYMIND_USE_SIN_37_91
    template<>
    struct SinTableValueSize<37, 91, true>
    {
        typedef SinValuesTableQ37_91 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_37_91

    #if TINYMIND_USE_SIN_38_90
    template<>
    struct SinTableValueSize<38, 90, true>
    {
        typedef SinValuesTableQ38_90 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_38_90

    #if TINYMIND_USE_SIN_39_89
    template<>
    struct SinTableValueSize<39, 89, true>
    {
        typedef SinValuesTableQ39_89 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_39_89

    #if TINYMIND_USE_SIN_40_88
    template<>
    struct SinTableValueSize<40, 88, true>
    {
        typedef SinValuesTableQ40_88 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_40_88

    #if TINYMIND_USE_SIN_41_87
    template<>
    struct SinTableValueSize<41, 87, true>
    {
        typedef SinValuesTableQ41_87 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_41_87

    #if TINYMIND_USE_SIN_42_86
    template<>
    struct SinTableValueSize<42, 86, true>
    {
        typedef SinValuesTableQ42_86 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_42_86

    #if TINYMIND_USE_SIN_43_85
    template<>
    struct SinTableValueSize<43, 85, true>
    {
        typedef SinValuesTableQ43_85 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_43_85

    #if TINYMIND_USE_SIN_44_84
    template<>
    struct SinTableValueSize<44, 84, true>
    {
        typedef SinValuesTableQ44_84 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_44_84

    #if TINYMIND_USE_SIN_45_83
    template<>
    struct SinTableValueSize<45, 83, true>
    {
        typedef SinValuesTableQ45_83 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_45_83

    #if TINYMIND_USE_SIN_46_82
    template<>
    struct SinTableValueSize<46, 82, true>
    {
        typedef SinValuesTableQ46_82 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_46_82

    #if TINYMIND_USE_SIN_47_81
    template<>
    struct SinTableValueSize<47, 81, true>
    {
        typedef SinValuesTableQ47_81 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_47_81

    #if TINYMIND_USE_SIN_48_80
    template<>
    struct SinTableValueSize<48, 80, true>
    {
        typedef SinValuesTableQ48_80 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_48_80

    #if TINYMIND_USE_SIN_49_79
    template<>
    struct SinTableValueSize<49, 79, true>
    {
        typedef SinValuesTableQ49_79 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_49_79

    #if TINYMIND_USE_SIN_50_78
    template<>
    struct SinTableValueSize<50, 78, true>
    {
        typedef SinValuesTableQ50_78 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_50_78

    #if TINYMIND_USE_SIN_51_77
    template<>
    struct SinTableValueSize<51, 77, true>
    {
        typedef SinValuesTableQ51_77 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_51_77

    #if TINYMIND_USE_SIN_52_76
    template<>
    struct SinTableValueSize<52, 76, true>
    {
        typedef SinValuesTableQ52_76 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_52_76

    #if TINYMIND_USE_SIN_53_75
    template<>
    struct SinTableValueSize<53, 75, true>
    {
        typedef SinValuesTableQ53_75 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_53_75

    #if TINYMIND_USE_SIN_54_74
    template<>
    struct SinTableValueSize<54, 74, true>
    {
        typedef SinValuesTableQ54_74 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_54_74

    #if TINYMIND_USE_SIN_55_73
    template<>
    struct SinTableValueSize<55, 73, true>
    {
        typedef SinValuesTableQ55_73 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_55_73

    #if TINYMIND_USE_SIN_56_72
    template<>
    struct SinTableValueSize<56, 72, true>
    {
        typedef SinValuesTableQ56_72 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_56_72

    #if TINYMIND_USE_SIN_57_71
    template<>
    struct SinTableValueSize<57, 71, true>
    {
        typedef SinValuesTableQ57_71 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_57_71

    #if TINYMIND_USE_SIN_58_70
    template<>
    struct SinTableValueSize<58, 70, true>
    {
        typedef SinValuesTableQ58_70 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_58_70

    #if TINYMIND_USE_SIN_59_69
    template<>
    struct SinTableValueSize<59, 69, true>
    {
        typedef SinValuesTableQ59_69 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_59_69

    #if TINYMIND_USE_SIN_60_68
    template<>
    struct SinTableValueSize<60, 68, true>
    {
        typedef SinValuesTableQ60_68 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_60_68

    #if TINYMIND_USE_SIN_61_67
    template<>
    struct SinTableValueSize<61, 67, true>
    {
        typedef SinValuesTableQ61_67 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_61_67

    #if TINYMIND_USE_SIN_62_66
    template<>
    struct SinTableValueSize<62, 66, true>
    {
        typedef SinValuesTableQ62_66 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_62_66

    #if TINYMIND_USE_SIN_63_65
    template<>
    struct SinTableValueSize<63, 65, true>
    {
        typedef SinValuesTableQ63_65 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_63_65

    #if TINYMIND_USE_SIN_64_64
    template<>
    struct SinTableValueSize<64, 64, true>
    {
        typedef SinValuesTableQ64_64 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_64_64

    #if TINYMIND_USE_SIN_65_63
    template<>
    struct SinTableValueSize<65, 63, true>
    {
        typedef SinValuesTableQ65_63 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_65_63

    #if TINYMIND_USE_SIN_66_62
    template<>
    struct SinTableValueSize<66, 62, true>
    {
        typedef SinValuesTableQ66_62 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_66_62

    #if TINYMIND_USE_SIN_67_61
    template<>
    struct SinTableValueSize<67, 61, true>
    {
        typedef SinValuesTableQ67_61 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_67_61

    #if TINYMIND_USE_SIN_68_60
    template<>
    struct SinTableValueSize<68, 60, true>
    {
        typedef SinValuesTableQ68_60 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_68_60

    #if TINYMIND_USE_SIN_69_59
    template<>
    struct SinTableValueSize<69, 59, true>
    {
        typedef SinValuesTableQ69_59 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_69_59

    #if TINYMIND_USE_SIN_70_58
    template<>
    struct SinTableValueSize<70, 58, true>
    {
        typedef SinValuesTableQ70_58 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_70_58

    #if TINYMIND_USE_SIN_71_57
    template<>
    struct SinTableValueSize<71, 57, true>
    {
        typedef SinValuesTableQ71_57 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_71_57

    #if TINYMIND_USE_SIN_72_56
    template<>
    struct SinTableValueSize<72, 56, true>
    {
        typedef SinValuesTableQ72_56 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_72_56

    #if TINYMIND_USE_SIN_73_55
    template<>
    struct SinTableValueSize<73, 55, true>
    {
        typedef SinValuesTableQ73_55 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_73_55

    #if TINYMIND_USE_SIN_74_54
    template<>
    struct SinTableValueSize<74, 54, true>
    {
        typedef SinValuesTableQ74_54 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_74_54

    #if TINYMIND_USE_SIN_75_53
    template<>
    struct SinTableValueSize<75, 53, true>
    {
        typedef SinValuesTableQ75_53 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_75_53

    #if TINYMIND_USE_SIN_76_52
    template<>
    struct SinTableValueSize<76, 52, true>
    {
        typedef SinValuesTableQ76_52 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_76_52

    #if TINYMIND_USE_SIN_77_51
    template<>
    struct SinTableValueSize<77, 51, true>
    {
        typedef SinValuesTableQ77_51 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_77_51

    #if TINYMIND_USE_SIN_78_50
    template<>
    struct SinTableValueSize<78, 50, true>
    {
        typedef SinValuesTableQ78_50 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_78_50

    #if TINYMIND_USE_SIN_79_49
    template<>
    struct SinTableValueSize<79, 49, true>
    {
        typedef SinValuesTableQ79_49 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_79_49

    #if TINYMIND_USE_SIN_80_48
    template<>
    struct SinTableValueSize<80, 48, true>
    {
        typedef SinValuesTableQ80_48 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_80_48

    #if TINYMIND_USE_SIN_81_47
    template<>
    struct SinTableValueSize<81, 47, true>
    {
        typedef SinValuesTableQ81_47 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_81_47

    #if TINYMIND_USE_SIN_82_46
    template<>
    struct SinTableValueSize<82, 46, true>
    {
        typedef SinValuesTableQ82_46 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_82_46

    #if TINYMIND_USE_SIN_83_45
    template<>
    struct SinTableValueSize<83, 45, true>
    {
        typedef SinValuesTableQ83_45 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_83_45

    #if TINYMIND_USE_SIN_84_44
    template<>
    struct SinTableValueSize<84, 44, true>
    {
        typedef SinValuesTableQ84_44 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_84_44

    #if TINYMIND_USE_SIN_85_43
    template<>
    struct SinTableValueSize<85, 43, true>
    {
        typedef SinValuesTableQ85_43 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_85_43

    #if TINYMIND_USE_SIN_86_42
    template<>
    struct SinTableValueSize<86, 42, true>
    {
        typedef SinValuesTableQ86_42 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_86_42

    #if TINYMIND_USE_SIN_87_41
    template<>
    struct SinTableValueSize<87, 41, true>
    {
        typedef SinValuesTableQ87_41 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_87_41

    #if TINYMIND_USE_SIN_88_40
    template<>
    struct SinTableValueSize<88, 40, true>
    {
        typedef SinValuesTableQ88_40 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_88_40

    #if TINYMIND_USE_SIN_89_39
    template<>
    struct SinTableValueSize<89, 39, true>
    {
        typedef SinValuesTableQ89_39 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_89_39

    #if TINYMIND_USE_SIN_90_38
    template<>
    struct SinTableValueSize<90, 38, true>
    {
        typedef SinValuesTableQ90_38 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_90_38

    #if TINYMIND_USE_SIN_91_37
    template<>
    struct SinTableValueSize<91, 37, true>
    {
        typedef SinValuesTableQ91_37 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_91_37

    #if TINYMIND_USE_SIN_92_36
    template<>
    struct SinTableValueSize<92, 36, true>
    {
        typedef SinValuesTableQ92_36 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_92_36

    #if TINYMIND_USE_SIN_93_35
    template<>
    struct SinTableValueSize<93, 35, true>
    {
        typedef SinValuesTableQ93_35 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_93_35

    #if TINYMIND_USE_SIN_94_34
    template<>
    struct SinTableValueSize<94, 34, true>
    {
        typedef SinValuesTableQ94_34 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_94_34

    #if TINYMIND_USE_SIN_95_33
    template<>
    struct SinTableValueSize<95, 33, true>
    {
        typedef SinValuesTableQ95_33 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_95_33

    #if TINYMIND_USE_SIN_96_32
    template<>
    struct SinTableValueSize<96, 32, true>
    {
        typedef SinValuesTableQ96_32 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_96_32

    #if TINYMIND_USE_SIN_97_31
    template<>
    struct SinTableValueSize<97, 31, true>
    {
        typedef SinValuesTableQ97_31 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_97_31

    #if TINYMIND_USE_SIN_98_30
    template<>
    struct SinTableValueSize<98, 30, true>
    {
        typedef SinValuesTableQ98_30 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_98_30

    #if TINYMIND_USE_SIN_99_29
    template<>
    struct SinTableValueSize<99, 29, true>
    {
        typedef SinValuesTableQ99_29 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_99_29

    #if TINYMIND_USE_SIN_100_28
    template<>
    struct SinTableValueSize<100, 28, true>
    {
        typedef SinValuesTableQ100_28 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_100_28

    #if TINYMIND_USE_SIN_101_27
    template<>
    struct SinTableValueSize<101, 27, true>
    {
        typedef SinValuesTableQ101_27 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_101_27

    #if TINYMIND_USE_SIN_102_26
    template<>
    struct SinTableValueSize<102, 26, true>
    {
        typedef SinValuesTableQ102_26 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_102_26

    #if TINYMIND_USE_SIN_103_25
    template<>
    struct SinTableValueSize<103, 25, true>
    {
        typedef SinValuesTableQ103_25 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_103_25

    #if TINYMIND_USE_SIN_104_24
    template<>
    struct SinTableValueSize<104, 24, true>
    {
        typedef SinValuesTableQ104_24 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_104_24

    #if TINYMIND_USE_SIN_105_23
    template<>
    struct SinTableValueSize<105, 23, true>
    {
        typedef SinValuesTableQ105_23 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_105_23

    #if TINYMIND_USE_SIN_106_22
    template<>
    struct SinTableValueSize<106, 22, true>
    {
        typedef SinValuesTableQ106_22 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_106_22

    #if TINYMIND_USE_SIN_107_21
    template<>
    struct SinTableValueSize<107, 21, true>
    {
        typedef SinValuesTableQ107_21 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_107_21

    #if TINYMIND_USE_SIN_108_20
    template<>
    struct SinTableValueSize<108, 20, true>
    {
        typedef SinValuesTableQ108_20 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_108_20

    #if TINYMIND_USE_SIN_109_19
    template<>
    struct SinTableValueSize<109, 19, true>
    {
        typedef SinValuesTableQ109_19 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_109_19

    #if TINYMIND_USE_SIN_110_18
    template<>
    struct SinTableValueSize<110, 18, true>
    {
        typedef SinValuesTableQ110_18 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_110_18

    #if TINYMIND_USE_SIN_111_17
    template<>
    struct SinTableValueSize<111, 17, true>
    {
        typedef SinValuesTableQ111_17 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_111_17

    #if TINYMIND_USE_SIN_112_16
    template<>
    struct SinTableValueSize<112, 16, true>
    {
        typedef SinValuesTableQ112_16 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_112_16

    #if TINYMIND_USE_SIN_113_15
    template<>
    struct SinTableValueSize<113, 15, true>
    {
        typedef SinValuesTableQ113_15 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_113_15

    #if TINYMIND_USE_SIN_114_14
    template<>
    struct SinTableValueSize<114, 14, true>
    {
        typedef SinValuesTableQ114_14 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_114_14

    #if TINYMIND_USE_SIN_115_13
    template<>
    struct SinTableValueSize<115, 13, true>
    {
        typedef SinValuesTableQ115_13 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_115_13

    #if TINYMIND_USE_SIN_116_12
    template<>
    struct SinTableValueSize<116, 12, true>
    {
        typedef SinValuesTableQ116_12 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_116_12

    #if TINYMIND_USE_SIN_117_11
    template<>
    struct SinTableValueSize<117, 11, true>
    {
        typedef SinValuesTableQ117_11 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_117_11

    #if TINYMIND_USE_SIN_118_10
    template<>
    struct SinTableValueSize<118, 10, true>
    {
        typedef SinValuesTableQ118_10 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_118_10

    #if TINYMIND_USE_SIN_119_9
    template<>
    struct SinTableValueSize<119, 9, true>
    {
        typedef SinValuesTableQ119_9 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_119_9

    #if TINYMIND_USE_SIN_120_8
    template<>
    struct SinTableValueSize<120, 8, true>
    {
        typedef SinValuesTableQ120_8 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_120_8

    #if TINYMIND_USE_SIN_121_7
    template<>
    struct SinTableValueSize<121, 7, true>
    {
        typedef SinValuesTableQ121_7 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_121_7

    #if TINYMIND_USE_SIN_122_6
    template<>
    struct SinTableValueSize<122, 6, true>
    {
        typedef SinValuesTableQ122_6 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_122_6

    #if TINYMIND_USE_SIN_123_5
    template<>
    struct SinTableValueSize<123, 5, true>
    {
        typedef SinValuesTableQ123_5 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_123_5

    #if TINYMIND_USE_SIN_124_4
    template<>
    struct SinTableValueSize<124, 4, true>
    {
        typedef SinValuesTableQ124_4 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_124_4

    #if TINYMIND_USE_SIN_125_3
    template<>
    struct SinTableValueSize<125, 3, true>
    {
        typedef SinValuesTableQ125_3 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_125_3

    #if TINYMIND_USE_SIN_126_2
    template<>
    struct SinTableValueSize<126, 2, true>
    {
        typedef SinValuesTableQ126_2 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_126_2

    #if TINYMIND_USE_SIN_127_1
    template<>
    struct SinTableValueSize<127, 1, true>
    {
        typedef SinValuesTableQ127_1 SinTableType;
    };
    #endif // TINYMIND_USE_SIN_127_1

    template<unsigned FixedBits,unsigned FracBits, bool IsSigned>
    struct SinValuesTableSelector
    {
        typedef typename SinTableValueSize<FixedBits, FracBits, IsSigned>::SinTableType SinTableType;
    };
}
