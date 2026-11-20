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

    template<unsigned FixedBits,unsigned FracBits, bool IsSigned>
    struct SinValuesTableSelector
    {
        typedef typename SinTableValueSize<FixedBits, FracBits, IsSigned>::SinTableType SinTableType;
    };
}
