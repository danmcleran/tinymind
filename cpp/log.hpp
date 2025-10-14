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

#include "logValues8Bit.hpp"
#include "logValues16Bit.hpp"
#include "logValues32Bit.hpp"
#include "logValues64Bit.hpp"
#include "logValues128Bit.hpp"

namespace tinymind {
    template<unsigned FixedBits, unsigned FracBits, bool IsSigned>
    struct LogTableValueSize
    {
    };

    #if TINYMIND_USE_LOG_1_7
    template<>
    struct LogTableValueSize<1, 7, true>
    {
        typedef LogValuesTableQ1_7 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_1_7

    #if TINYMIND_USE_LOG_2_6
    template<>
    struct LogTableValueSize<2, 6, true>
    {
        typedef LogValuesTableQ2_6 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_2_6

    #if TINYMIND_USE_LOG_3_5
    template<>
    struct LogTableValueSize<3, 5, true>
    {
        typedef LogValuesTableQ3_5 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_3_5

    #if TINYMIND_USE_LOG_4_4
    template<>
    struct LogTableValueSize<4, 4, true>
    {
        typedef LogValuesTableQ4_4 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_4_4

    #if TINYMIND_USE_LOG_5_3
    template<>
    struct LogTableValueSize<5, 3, true>
    {
        typedef LogValuesTableQ5_3 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_5_3

    #if TINYMIND_USE_LOG_6_2
    template<>
    struct LogTableValueSize<6, 2, true>
    {
        typedef LogValuesTableQ6_2 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_6_2

    #if TINYMIND_USE_LOG_7_1
    template<>
    struct LogTableValueSize<7, 1, true>
    {
        typedef LogValuesTableQ7_1 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_7_1

    #if TINYMIND_USE_LOG_1_15
    template<>
    struct LogTableValueSize<1, 15, true>
    {
        typedef LogValuesTableQ1_15 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_1_15

    #if TINYMIND_USE_LOG_2_14
    template<>
    struct LogTableValueSize<2, 14, true>
    {
        typedef LogValuesTableQ2_14 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_2_14

    #if TINYMIND_USE_LOG_3_13
    template<>
    struct LogTableValueSize<3, 13, true>
    {
        typedef LogValuesTableQ3_13 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_3_13

    #if TINYMIND_USE_LOG_4_12
    template<>
    struct LogTableValueSize<4, 12, true>
    {
        typedef LogValuesTableQ4_12 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_4_12

    #if TINYMIND_USE_LOG_5_11
    template<>
    struct LogTableValueSize<5, 11, true>
    {
        typedef LogValuesTableQ5_11 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_5_11

    #if TINYMIND_USE_LOG_6_10
    template<>
    struct LogTableValueSize<6, 10, true>
    {
        typedef LogValuesTableQ6_10 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_6_10

    #if TINYMIND_USE_LOG_7_9
    template<>
    struct LogTableValueSize<7, 9, true>
    {
        typedef LogValuesTableQ7_9 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_7_9

    #if TINYMIND_USE_LOG_8_8
    template<>
    struct LogTableValueSize<8, 8, true>
    {
        typedef LogValuesTableQ8_8 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_8_8

    #if TINYMIND_USE_LOG_9_7
    template<>
    struct LogTableValueSize<9, 7, true>
    {
        typedef LogValuesTableQ9_7 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_9_7

    #if TINYMIND_USE_LOG_10_6
    template<>
    struct LogTableValueSize<10, 6, true>
    {
        typedef LogValuesTableQ10_6 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_10_6

    #if TINYMIND_USE_LOG_11_5
    template<>
    struct LogTableValueSize<11, 5, true>
    {
        typedef LogValuesTableQ11_5 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_11_5

    #if TINYMIND_USE_LOG_12_4
    template<>
    struct LogTableValueSize<12, 4, true>
    {
        typedef LogValuesTableQ12_4 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_12_4

    #if TINYMIND_USE_LOG_13_3
    template<>
    struct LogTableValueSize<13, 3, true>
    {
        typedef LogValuesTableQ13_3 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_13_3

    #if TINYMIND_USE_LOG_14_2
    template<>
    struct LogTableValueSize<14, 2, true>
    {
        typedef LogValuesTableQ14_2 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_14_2

    #if TINYMIND_USE_LOG_15_1
    template<>
    struct LogTableValueSize<15, 1, true>
    {
        typedef LogValuesTableQ15_1 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_15_1

    #if TINYMIND_USE_LOG_1_31
    template<>
    struct LogTableValueSize<1, 31, true>
    {
        typedef LogValuesTableQ1_31 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_1_31

    #if TINYMIND_USE_LOG_2_30
    template<>
    struct LogTableValueSize<2, 30, true>
    {
        typedef LogValuesTableQ2_30 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_2_30

    #if TINYMIND_USE_LOG_3_29
    template<>
    struct LogTableValueSize<3, 29, true>
    {
        typedef LogValuesTableQ3_29 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_3_29

    #if TINYMIND_USE_LOG_4_28
    template<>
    struct LogTableValueSize<4, 28, true>
    {
        typedef LogValuesTableQ4_28 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_4_28

    #if TINYMIND_USE_LOG_5_27
    template<>
    struct LogTableValueSize<5, 27, true>
    {
        typedef LogValuesTableQ5_27 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_5_27

    #if TINYMIND_USE_LOG_6_26
    template<>
    struct LogTableValueSize<6, 26, true>
    {
        typedef LogValuesTableQ6_26 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_6_26

    #if TINYMIND_USE_LOG_7_25
    template<>
    struct LogTableValueSize<7, 25, true>
    {
        typedef LogValuesTableQ7_25 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_7_25

    #if TINYMIND_USE_LOG_8_24
    template<>
    struct LogTableValueSize<8, 24, true>
    {
        typedef LogValuesTableQ8_24 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_8_24

    #if TINYMIND_USE_LOG_9_23
    template<>
    struct LogTableValueSize<9, 23, true>
    {
        typedef LogValuesTableQ9_23 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_9_23

    #if TINYMIND_USE_LOG_10_22
    template<>
    struct LogTableValueSize<10, 22, true>
    {
        typedef LogValuesTableQ10_22 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_10_22

    #if TINYMIND_USE_LOG_11_21
    template<>
    struct LogTableValueSize<11, 21, true>
    {
        typedef LogValuesTableQ11_21 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_11_21

    #if TINYMIND_USE_LOG_12_20
    template<>
    struct LogTableValueSize<12, 20, true>
    {
        typedef LogValuesTableQ12_20 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_12_20

    #if TINYMIND_USE_LOG_13_19
    template<>
    struct LogTableValueSize<13, 19, true>
    {
        typedef LogValuesTableQ13_19 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_13_19

    #if TINYMIND_USE_LOG_14_18
    template<>
    struct LogTableValueSize<14, 18, true>
    {
        typedef LogValuesTableQ14_18 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_14_18

    #if TINYMIND_USE_LOG_15_17
    template<>
    struct LogTableValueSize<15, 17, true>
    {
        typedef LogValuesTableQ15_17 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_15_17

    #if TINYMIND_USE_LOG_16_16
    template<>
    struct LogTableValueSize<16, 16, true>
    {
        typedef LogValuesTableQ16_16 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_16_16

    #if TINYMIND_USE_LOG_17_15
    template<>
    struct LogTableValueSize<17, 15, true>
    {
        typedef LogValuesTableQ17_15 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_17_15

    #if TINYMIND_USE_LOG_18_14
    template<>
    struct LogTableValueSize<18, 14, true>
    {
        typedef LogValuesTableQ18_14 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_18_14

    #if TINYMIND_USE_LOG_19_13
    template<>
    struct LogTableValueSize<19, 13, true>
    {
        typedef LogValuesTableQ19_13 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_19_13

    #if TINYMIND_USE_LOG_20_12
    template<>
    struct LogTableValueSize<20, 12, true>
    {
        typedef LogValuesTableQ20_12 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_20_12

    #if TINYMIND_USE_LOG_21_11
    template<>
    struct LogTableValueSize<21, 11, true>
    {
        typedef LogValuesTableQ21_11 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_21_11

    #if TINYMIND_USE_LOG_22_10
    template<>
    struct LogTableValueSize<22, 10, true>
    {
        typedef LogValuesTableQ22_10 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_22_10

    #if TINYMIND_USE_LOG_23_9
    template<>
    struct LogTableValueSize<23, 9, true>
    {
        typedef LogValuesTableQ23_9 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_23_9

    #if TINYMIND_USE_LOG_24_8
    template<>
    struct LogTableValueSize<24, 8, true>
    {
        typedef LogValuesTableQ24_8 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_24_8

    #if TINYMIND_USE_LOG_25_7
    template<>
    struct LogTableValueSize<25, 7, true>
    {
        typedef LogValuesTableQ25_7 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_25_7

    #if TINYMIND_USE_LOG_26_6
    template<>
    struct LogTableValueSize<26, 6, true>
    {
        typedef LogValuesTableQ26_6 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_26_6

    #if TINYMIND_USE_LOG_27_5
    template<>
    struct LogTableValueSize<27, 5, true>
    {
        typedef LogValuesTableQ27_5 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_27_5

    #if TINYMIND_USE_LOG_28_4
    template<>
    struct LogTableValueSize<28, 4, true>
    {
        typedef LogValuesTableQ28_4 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_28_4

    #if TINYMIND_USE_LOG_29_3
    template<>
    struct LogTableValueSize<29, 3, true>
    {
        typedef LogValuesTableQ29_3 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_29_3

    #if TINYMIND_USE_LOG_30_2
    template<>
    struct LogTableValueSize<30, 2, true>
    {
        typedef LogValuesTableQ30_2 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_30_2

    #if TINYMIND_USE_LOG_31_1
    template<>
    struct LogTableValueSize<31, 1, true>
    {
        typedef LogValuesTableQ31_1 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_31_1

    #if TINYMIND_USE_LOG_1_63
    template<>
    struct LogTableValueSize<1, 63, true>
    {
        typedef LogValuesTableQ1_63 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_1_63

    #if TINYMIND_USE_LOG_2_62
    template<>
    struct LogTableValueSize<2, 62, true>
    {
        typedef LogValuesTableQ2_62 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_2_62

    #if TINYMIND_USE_LOG_3_61
    template<>
    struct LogTableValueSize<3, 61, true>
    {
        typedef LogValuesTableQ3_61 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_3_61

    #if TINYMIND_USE_LOG_4_60
    template<>
    struct LogTableValueSize<4, 60, true>
    {
        typedef LogValuesTableQ4_60 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_4_60

    #if TINYMIND_USE_LOG_5_59
    template<>
    struct LogTableValueSize<5, 59, true>
    {
        typedef LogValuesTableQ5_59 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_5_59

    #if TINYMIND_USE_LOG_6_58
    template<>
    struct LogTableValueSize<6, 58, true>
    {
        typedef LogValuesTableQ6_58 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_6_58

    #if TINYMIND_USE_LOG_7_57
    template<>
    struct LogTableValueSize<7, 57, true>
    {
        typedef LogValuesTableQ7_57 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_7_57

    #if TINYMIND_USE_LOG_8_56
    template<>
    struct LogTableValueSize<8, 56, true>
    {
        typedef LogValuesTableQ8_56 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_8_56

    #if TINYMIND_USE_LOG_9_55
    template<>
    struct LogTableValueSize<9, 55, true>
    {
        typedef LogValuesTableQ9_55 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_9_55

    #if TINYMIND_USE_LOG_10_54
    template<>
    struct LogTableValueSize<10, 54, true>
    {
        typedef LogValuesTableQ10_54 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_10_54

    #if TINYMIND_USE_LOG_11_53
    template<>
    struct LogTableValueSize<11, 53, true>
    {
        typedef LogValuesTableQ11_53 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_11_53

    #if TINYMIND_USE_LOG_12_52
    template<>
    struct LogTableValueSize<12, 52, true>
    {
        typedef LogValuesTableQ12_52 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_12_52

    #if TINYMIND_USE_LOG_13_51
    template<>
    struct LogTableValueSize<13, 51, true>
    {
        typedef LogValuesTableQ13_51 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_13_51

    #if TINYMIND_USE_LOG_14_50
    template<>
    struct LogTableValueSize<14, 50, true>
    {
        typedef LogValuesTableQ14_50 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_14_50

    #if TINYMIND_USE_LOG_15_49
    template<>
    struct LogTableValueSize<15, 49, true>
    {
        typedef LogValuesTableQ15_49 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_15_49

    #if TINYMIND_USE_LOG_16_48
    template<>
    struct LogTableValueSize<16, 48, true>
    {
        typedef LogValuesTableQ16_48 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_16_48

    #if TINYMIND_USE_LOG_17_47
    template<>
    struct LogTableValueSize<17, 47, true>
    {
        typedef LogValuesTableQ17_47 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_17_47

    #if TINYMIND_USE_LOG_18_46
    template<>
    struct LogTableValueSize<18, 46, true>
    {
        typedef LogValuesTableQ18_46 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_18_46

    #if TINYMIND_USE_LOG_19_45
    template<>
    struct LogTableValueSize<19, 45, true>
    {
        typedef LogValuesTableQ19_45 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_19_45

    #if TINYMIND_USE_LOG_20_44
    template<>
    struct LogTableValueSize<20, 44, true>
    {
        typedef LogValuesTableQ20_44 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_20_44

    #if TINYMIND_USE_LOG_21_43
    template<>
    struct LogTableValueSize<21, 43, true>
    {
        typedef LogValuesTableQ21_43 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_21_43

    #if TINYMIND_USE_LOG_22_42
    template<>
    struct LogTableValueSize<22, 42, true>
    {
        typedef LogValuesTableQ22_42 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_22_42

    #if TINYMIND_USE_LOG_23_41
    template<>
    struct LogTableValueSize<23, 41, true>
    {
        typedef LogValuesTableQ23_41 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_23_41

    #if TINYMIND_USE_LOG_24_40
    template<>
    struct LogTableValueSize<24, 40, true>
    {
        typedef LogValuesTableQ24_40 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_24_40

    #if TINYMIND_USE_LOG_25_39
    template<>
    struct LogTableValueSize<25, 39, true>
    {
        typedef LogValuesTableQ25_39 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_25_39

    #if TINYMIND_USE_LOG_26_38
    template<>
    struct LogTableValueSize<26, 38, true>
    {
        typedef LogValuesTableQ26_38 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_26_38

    #if TINYMIND_USE_LOG_27_37
    template<>
    struct LogTableValueSize<27, 37, true>
    {
        typedef LogValuesTableQ27_37 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_27_37

    #if TINYMIND_USE_LOG_28_36
    template<>
    struct LogTableValueSize<28, 36, true>
    {
        typedef LogValuesTableQ28_36 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_28_36

    #if TINYMIND_USE_LOG_29_35
    template<>
    struct LogTableValueSize<29, 35, true>
    {
        typedef LogValuesTableQ29_35 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_29_35

    #if TINYMIND_USE_LOG_30_34
    template<>
    struct LogTableValueSize<30, 34, true>
    {
        typedef LogValuesTableQ30_34 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_30_34

    #if TINYMIND_USE_LOG_31_33
    template<>
    struct LogTableValueSize<31, 33, true>
    {
        typedef LogValuesTableQ31_33 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_31_33

    #if TINYMIND_USE_LOG_32_32
    template<>
    struct LogTableValueSize<32, 32, true>
    {
        typedef LogValuesTableQ32_32 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_32_32

    #if TINYMIND_USE_LOG_33_31
    template<>
    struct LogTableValueSize<33, 31, true>
    {
        typedef LogValuesTableQ33_31 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_33_31

    #if TINYMIND_USE_LOG_34_30
    template<>
    struct LogTableValueSize<34, 30, true>
    {
        typedef LogValuesTableQ34_30 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_34_30

    #if TINYMIND_USE_LOG_35_29
    template<>
    struct LogTableValueSize<35, 29, true>
    {
        typedef LogValuesTableQ35_29 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_35_29

    #if TINYMIND_USE_LOG_36_28
    template<>
    struct LogTableValueSize<36, 28, true>
    {
        typedef LogValuesTableQ36_28 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_36_28

    #if TINYMIND_USE_LOG_37_27
    template<>
    struct LogTableValueSize<37, 27, true>
    {
        typedef LogValuesTableQ37_27 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_37_27

    #if TINYMIND_USE_LOG_38_26
    template<>
    struct LogTableValueSize<38, 26, true>
    {
        typedef LogValuesTableQ38_26 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_38_26

    #if TINYMIND_USE_LOG_39_25
    template<>
    struct LogTableValueSize<39, 25, true>
    {
        typedef LogValuesTableQ39_25 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_39_25

    #if TINYMIND_USE_LOG_40_24
    template<>
    struct LogTableValueSize<40, 24, true>
    {
        typedef LogValuesTableQ40_24 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_40_24

    #if TINYMIND_USE_LOG_41_23
    template<>
    struct LogTableValueSize<41, 23, true>
    {
        typedef LogValuesTableQ41_23 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_41_23

    #if TINYMIND_USE_LOG_42_22
    template<>
    struct LogTableValueSize<42, 22, true>
    {
        typedef LogValuesTableQ42_22 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_42_22

    #if TINYMIND_USE_LOG_43_21
    template<>
    struct LogTableValueSize<43, 21, true>
    {
        typedef LogValuesTableQ43_21 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_43_21

    #if TINYMIND_USE_LOG_44_20
    template<>
    struct LogTableValueSize<44, 20, true>
    {
        typedef LogValuesTableQ44_20 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_44_20

    #if TINYMIND_USE_LOG_45_19
    template<>
    struct LogTableValueSize<45, 19, true>
    {
        typedef LogValuesTableQ45_19 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_45_19

    #if TINYMIND_USE_LOG_46_18
    template<>
    struct LogTableValueSize<46, 18, true>
    {
        typedef LogValuesTableQ46_18 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_46_18

    #if TINYMIND_USE_LOG_47_17
    template<>
    struct LogTableValueSize<47, 17, true>
    {
        typedef LogValuesTableQ47_17 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_47_17

    #if TINYMIND_USE_LOG_48_16
    template<>
    struct LogTableValueSize<48, 16, true>
    {
        typedef LogValuesTableQ48_16 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_48_16

    #if TINYMIND_USE_LOG_49_15
    template<>
    struct LogTableValueSize<49, 15, true>
    {
        typedef LogValuesTableQ49_15 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_49_15

    #if TINYMIND_USE_LOG_50_14
    template<>
    struct LogTableValueSize<50, 14, true>
    {
        typedef LogValuesTableQ50_14 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_50_14

    #if TINYMIND_USE_LOG_51_13
    template<>
    struct LogTableValueSize<51, 13, true>
    {
        typedef LogValuesTableQ51_13 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_51_13

    #if TINYMIND_USE_LOG_52_12
    template<>
    struct LogTableValueSize<52, 12, true>
    {
        typedef LogValuesTableQ52_12 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_52_12

    #if TINYMIND_USE_LOG_53_11
    template<>
    struct LogTableValueSize<53, 11, true>
    {
        typedef LogValuesTableQ53_11 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_53_11

    #if TINYMIND_USE_LOG_54_10
    template<>
    struct LogTableValueSize<54, 10, true>
    {
        typedef LogValuesTableQ54_10 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_54_10

    #if TINYMIND_USE_LOG_55_9
    template<>
    struct LogTableValueSize<55, 9, true>
    {
        typedef LogValuesTableQ55_9 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_55_9

    #if TINYMIND_USE_LOG_56_8
    template<>
    struct LogTableValueSize<56, 8, true>
    {
        typedef LogValuesTableQ56_8 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_56_8

    #if TINYMIND_USE_LOG_57_7
    template<>
    struct LogTableValueSize<57, 7, true>
    {
        typedef LogValuesTableQ57_7 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_57_7

    #if TINYMIND_USE_LOG_58_6
    template<>
    struct LogTableValueSize<58, 6, true>
    {
        typedef LogValuesTableQ58_6 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_58_6

    #if TINYMIND_USE_LOG_59_5
    template<>
    struct LogTableValueSize<59, 5, true>
    {
        typedef LogValuesTableQ59_5 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_59_5

    #if TINYMIND_USE_LOG_60_4
    template<>
    struct LogTableValueSize<60, 4, true>
    {
        typedef LogValuesTableQ60_4 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_60_4

    #if TINYMIND_USE_LOG_61_3
    template<>
    struct LogTableValueSize<61, 3, true>
    {
        typedef LogValuesTableQ61_3 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_61_3

    #if TINYMIND_USE_LOG_62_2
    template<>
    struct LogTableValueSize<62, 2, true>
    {
        typedef LogValuesTableQ62_2 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_62_2

    #if TINYMIND_USE_LOG_63_1
    template<>
    struct LogTableValueSize<63, 1, true>
    {
        typedef LogValuesTableQ63_1 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_63_1

    #if TINYMIND_USE_LOG_1_127
    template<>
    struct LogTableValueSize<1, 127, true>
    {
        typedef LogValuesTableQ1_127 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_1_127

    #if TINYMIND_USE_LOG_2_126
    template<>
    struct LogTableValueSize<2, 126, true>
    {
        typedef LogValuesTableQ2_126 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_2_126

    #if TINYMIND_USE_LOG_3_125
    template<>
    struct LogTableValueSize<3, 125, true>
    {
        typedef LogValuesTableQ3_125 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_3_125

    #if TINYMIND_USE_LOG_4_124
    template<>
    struct LogTableValueSize<4, 124, true>
    {
        typedef LogValuesTableQ4_124 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_4_124

    #if TINYMIND_USE_LOG_5_123
    template<>
    struct LogTableValueSize<5, 123, true>
    {
        typedef LogValuesTableQ5_123 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_5_123

    #if TINYMIND_USE_LOG_6_122
    template<>
    struct LogTableValueSize<6, 122, true>
    {
        typedef LogValuesTableQ6_122 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_6_122

    #if TINYMIND_USE_LOG_7_121
    template<>
    struct LogTableValueSize<7, 121, true>
    {
        typedef LogValuesTableQ7_121 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_7_121

    #if TINYMIND_USE_LOG_8_120
    template<>
    struct LogTableValueSize<8, 120, true>
    {
        typedef LogValuesTableQ8_120 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_8_120

    #if TINYMIND_USE_LOG_9_119
    template<>
    struct LogTableValueSize<9, 119, true>
    {
        typedef LogValuesTableQ9_119 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_9_119

    #if TINYMIND_USE_LOG_10_118
    template<>
    struct LogTableValueSize<10, 118, true>
    {
        typedef LogValuesTableQ10_118 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_10_118

    #if TINYMIND_USE_LOG_11_117
    template<>
    struct LogTableValueSize<11, 117, true>
    {
        typedef LogValuesTableQ11_117 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_11_117

    #if TINYMIND_USE_LOG_12_116
    template<>
    struct LogTableValueSize<12, 116, true>
    {
        typedef LogValuesTableQ12_116 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_12_116

    #if TINYMIND_USE_LOG_13_115
    template<>
    struct LogTableValueSize<13, 115, true>
    {
        typedef LogValuesTableQ13_115 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_13_115

    #if TINYMIND_USE_LOG_14_114
    template<>
    struct LogTableValueSize<14, 114, true>
    {
        typedef LogValuesTableQ14_114 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_14_114

    #if TINYMIND_USE_LOG_15_113
    template<>
    struct LogTableValueSize<15, 113, true>
    {
        typedef LogValuesTableQ15_113 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_15_113

    #if TINYMIND_USE_LOG_16_112
    template<>
    struct LogTableValueSize<16, 112, true>
    {
        typedef LogValuesTableQ16_112 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_16_112

    #if TINYMIND_USE_LOG_17_111
    template<>
    struct LogTableValueSize<17, 111, true>
    {
        typedef LogValuesTableQ17_111 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_17_111

    #if TINYMIND_USE_LOG_18_110
    template<>
    struct LogTableValueSize<18, 110, true>
    {
        typedef LogValuesTableQ18_110 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_18_110

    #if TINYMIND_USE_LOG_19_109
    template<>
    struct LogTableValueSize<19, 109, true>
    {
        typedef LogValuesTableQ19_109 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_19_109

    #if TINYMIND_USE_LOG_20_108
    template<>
    struct LogTableValueSize<20, 108, true>
    {
        typedef LogValuesTableQ20_108 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_20_108

    #if TINYMIND_USE_LOG_21_107
    template<>
    struct LogTableValueSize<21, 107, true>
    {
        typedef LogValuesTableQ21_107 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_21_107

    #if TINYMIND_USE_LOG_22_106
    template<>
    struct LogTableValueSize<22, 106, true>
    {
        typedef LogValuesTableQ22_106 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_22_106

    #if TINYMIND_USE_LOG_23_105
    template<>
    struct LogTableValueSize<23, 105, true>
    {
        typedef LogValuesTableQ23_105 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_23_105

    #if TINYMIND_USE_LOG_24_104
    template<>
    struct LogTableValueSize<24, 104, true>
    {
        typedef LogValuesTableQ24_104 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_24_104

    #if TINYMIND_USE_LOG_25_103
    template<>
    struct LogTableValueSize<25, 103, true>
    {
        typedef LogValuesTableQ25_103 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_25_103

    #if TINYMIND_USE_LOG_26_102
    template<>
    struct LogTableValueSize<26, 102, true>
    {
        typedef LogValuesTableQ26_102 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_26_102

    #if TINYMIND_USE_LOG_27_101
    template<>
    struct LogTableValueSize<27, 101, true>
    {
        typedef LogValuesTableQ27_101 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_27_101

    #if TINYMIND_USE_LOG_28_100
    template<>
    struct LogTableValueSize<28, 100, true>
    {
        typedef LogValuesTableQ28_100 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_28_100

    #if TINYMIND_USE_LOG_29_99
    template<>
    struct LogTableValueSize<29, 99, true>
    {
        typedef LogValuesTableQ29_99 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_29_99

    #if TINYMIND_USE_LOG_30_98
    template<>
    struct LogTableValueSize<30, 98, true>
    {
        typedef LogValuesTableQ30_98 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_30_98

    #if TINYMIND_USE_LOG_31_97
    template<>
    struct LogTableValueSize<31, 97, true>
    {
        typedef LogValuesTableQ31_97 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_31_97

    #if TINYMIND_USE_LOG_32_96
    template<>
    struct LogTableValueSize<32, 96, true>
    {
        typedef LogValuesTableQ32_96 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_32_96

    #if TINYMIND_USE_LOG_33_95
    template<>
    struct LogTableValueSize<33, 95, true>
    {
        typedef LogValuesTableQ33_95 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_33_95

    #if TINYMIND_USE_LOG_34_94
    template<>
    struct LogTableValueSize<34, 94, true>
    {
        typedef LogValuesTableQ34_94 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_34_94

    #if TINYMIND_USE_LOG_35_93
    template<>
    struct LogTableValueSize<35, 93, true>
    {
        typedef LogValuesTableQ35_93 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_35_93

    #if TINYMIND_USE_LOG_36_92
    template<>
    struct LogTableValueSize<36, 92, true>
    {
        typedef LogValuesTableQ36_92 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_36_92

    #if TINYMIND_USE_LOG_37_91
    template<>
    struct LogTableValueSize<37, 91, true>
    {
        typedef LogValuesTableQ37_91 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_37_91

    #if TINYMIND_USE_LOG_38_90
    template<>
    struct LogTableValueSize<38, 90, true>
    {
        typedef LogValuesTableQ38_90 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_38_90

    #if TINYMIND_USE_LOG_39_89
    template<>
    struct LogTableValueSize<39, 89, true>
    {
        typedef LogValuesTableQ39_89 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_39_89

    #if TINYMIND_USE_LOG_40_88
    template<>
    struct LogTableValueSize<40, 88, true>
    {
        typedef LogValuesTableQ40_88 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_40_88

    #if TINYMIND_USE_LOG_41_87
    template<>
    struct LogTableValueSize<41, 87, true>
    {
        typedef LogValuesTableQ41_87 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_41_87

    #if TINYMIND_USE_LOG_42_86
    template<>
    struct LogTableValueSize<42, 86, true>
    {
        typedef LogValuesTableQ42_86 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_42_86

    #if TINYMIND_USE_LOG_43_85
    template<>
    struct LogTableValueSize<43, 85, true>
    {
        typedef LogValuesTableQ43_85 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_43_85

    #if TINYMIND_USE_LOG_44_84
    template<>
    struct LogTableValueSize<44, 84, true>
    {
        typedef LogValuesTableQ44_84 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_44_84

    #if TINYMIND_USE_LOG_45_83
    template<>
    struct LogTableValueSize<45, 83, true>
    {
        typedef LogValuesTableQ45_83 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_45_83

    #if TINYMIND_USE_LOG_46_82
    template<>
    struct LogTableValueSize<46, 82, true>
    {
        typedef LogValuesTableQ46_82 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_46_82

    #if TINYMIND_USE_LOG_47_81
    template<>
    struct LogTableValueSize<47, 81, true>
    {
        typedef LogValuesTableQ47_81 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_47_81

    #if TINYMIND_USE_LOG_48_80
    template<>
    struct LogTableValueSize<48, 80, true>
    {
        typedef LogValuesTableQ48_80 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_48_80

    #if TINYMIND_USE_LOG_49_79
    template<>
    struct LogTableValueSize<49, 79, true>
    {
        typedef LogValuesTableQ49_79 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_49_79

    #if TINYMIND_USE_LOG_50_78
    template<>
    struct LogTableValueSize<50, 78, true>
    {
        typedef LogValuesTableQ50_78 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_50_78

    #if TINYMIND_USE_LOG_51_77
    template<>
    struct LogTableValueSize<51, 77, true>
    {
        typedef LogValuesTableQ51_77 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_51_77

    #if TINYMIND_USE_LOG_52_76
    template<>
    struct LogTableValueSize<52, 76, true>
    {
        typedef LogValuesTableQ52_76 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_52_76

    #if TINYMIND_USE_LOG_53_75
    template<>
    struct LogTableValueSize<53, 75, true>
    {
        typedef LogValuesTableQ53_75 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_53_75

    #if TINYMIND_USE_LOG_54_74
    template<>
    struct LogTableValueSize<54, 74, true>
    {
        typedef LogValuesTableQ54_74 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_54_74

    #if TINYMIND_USE_LOG_55_73
    template<>
    struct LogTableValueSize<55, 73, true>
    {
        typedef LogValuesTableQ55_73 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_55_73

    #if TINYMIND_USE_LOG_56_72
    template<>
    struct LogTableValueSize<56, 72, true>
    {
        typedef LogValuesTableQ56_72 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_56_72

    #if TINYMIND_USE_LOG_57_71
    template<>
    struct LogTableValueSize<57, 71, true>
    {
        typedef LogValuesTableQ57_71 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_57_71

    #if TINYMIND_USE_LOG_58_70
    template<>
    struct LogTableValueSize<58, 70, true>
    {
        typedef LogValuesTableQ58_70 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_58_70

    #if TINYMIND_USE_LOG_59_69
    template<>
    struct LogTableValueSize<59, 69, true>
    {
        typedef LogValuesTableQ59_69 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_59_69

    #if TINYMIND_USE_LOG_60_68
    template<>
    struct LogTableValueSize<60, 68, true>
    {
        typedef LogValuesTableQ60_68 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_60_68

    #if TINYMIND_USE_LOG_61_67
    template<>
    struct LogTableValueSize<61, 67, true>
    {
        typedef LogValuesTableQ61_67 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_61_67

    #if TINYMIND_USE_LOG_62_66
    template<>
    struct LogTableValueSize<62, 66, true>
    {
        typedef LogValuesTableQ62_66 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_62_66

    #if TINYMIND_USE_LOG_63_65
    template<>
    struct LogTableValueSize<63, 65, true>
    {
        typedef LogValuesTableQ63_65 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_63_65

    #if TINYMIND_USE_LOG_64_64
    template<>
    struct LogTableValueSize<64, 64, true>
    {
        typedef LogValuesTableQ64_64 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_64_64

    #if TINYMIND_USE_LOG_65_63
    template<>
    struct LogTableValueSize<65, 63, true>
    {
        typedef LogValuesTableQ65_63 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_65_63

    #if TINYMIND_USE_LOG_66_62
    template<>
    struct LogTableValueSize<66, 62, true>
    {
        typedef LogValuesTableQ66_62 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_66_62

    #if TINYMIND_USE_LOG_67_61
    template<>
    struct LogTableValueSize<67, 61, true>
    {
        typedef LogValuesTableQ67_61 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_67_61

    #if TINYMIND_USE_LOG_68_60
    template<>
    struct LogTableValueSize<68, 60, true>
    {
        typedef LogValuesTableQ68_60 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_68_60

    #if TINYMIND_USE_LOG_69_59
    template<>
    struct LogTableValueSize<69, 59, true>
    {
        typedef LogValuesTableQ69_59 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_69_59

    #if TINYMIND_USE_LOG_70_58
    template<>
    struct LogTableValueSize<70, 58, true>
    {
        typedef LogValuesTableQ70_58 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_70_58

    #if TINYMIND_USE_LOG_71_57
    template<>
    struct LogTableValueSize<71, 57, true>
    {
        typedef LogValuesTableQ71_57 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_71_57

    #if TINYMIND_USE_LOG_72_56
    template<>
    struct LogTableValueSize<72, 56, true>
    {
        typedef LogValuesTableQ72_56 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_72_56

    #if TINYMIND_USE_LOG_73_55
    template<>
    struct LogTableValueSize<73, 55, true>
    {
        typedef LogValuesTableQ73_55 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_73_55

    #if TINYMIND_USE_LOG_74_54
    template<>
    struct LogTableValueSize<74, 54, true>
    {
        typedef LogValuesTableQ74_54 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_74_54

    #if TINYMIND_USE_LOG_75_53
    template<>
    struct LogTableValueSize<75, 53, true>
    {
        typedef LogValuesTableQ75_53 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_75_53

    #if TINYMIND_USE_LOG_76_52
    template<>
    struct LogTableValueSize<76, 52, true>
    {
        typedef LogValuesTableQ76_52 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_76_52

    #if TINYMIND_USE_LOG_77_51
    template<>
    struct LogTableValueSize<77, 51, true>
    {
        typedef LogValuesTableQ77_51 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_77_51

    #if TINYMIND_USE_LOG_78_50
    template<>
    struct LogTableValueSize<78, 50, true>
    {
        typedef LogValuesTableQ78_50 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_78_50

    #if TINYMIND_USE_LOG_79_49
    template<>
    struct LogTableValueSize<79, 49, true>
    {
        typedef LogValuesTableQ79_49 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_79_49

    #if TINYMIND_USE_LOG_80_48
    template<>
    struct LogTableValueSize<80, 48, true>
    {
        typedef LogValuesTableQ80_48 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_80_48

    #if TINYMIND_USE_LOG_81_47
    template<>
    struct LogTableValueSize<81, 47, true>
    {
        typedef LogValuesTableQ81_47 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_81_47

    #if TINYMIND_USE_LOG_82_46
    template<>
    struct LogTableValueSize<82, 46, true>
    {
        typedef LogValuesTableQ82_46 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_82_46

    #if TINYMIND_USE_LOG_83_45
    template<>
    struct LogTableValueSize<83, 45, true>
    {
        typedef LogValuesTableQ83_45 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_83_45

    #if TINYMIND_USE_LOG_84_44
    template<>
    struct LogTableValueSize<84, 44, true>
    {
        typedef LogValuesTableQ84_44 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_84_44

    #if TINYMIND_USE_LOG_85_43
    template<>
    struct LogTableValueSize<85, 43, true>
    {
        typedef LogValuesTableQ85_43 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_85_43

    #if TINYMIND_USE_LOG_86_42
    template<>
    struct LogTableValueSize<86, 42, true>
    {
        typedef LogValuesTableQ86_42 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_86_42

    #if TINYMIND_USE_LOG_87_41
    template<>
    struct LogTableValueSize<87, 41, true>
    {
        typedef LogValuesTableQ87_41 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_87_41

    #if TINYMIND_USE_LOG_88_40
    template<>
    struct LogTableValueSize<88, 40, true>
    {
        typedef LogValuesTableQ88_40 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_88_40

    #if TINYMIND_USE_LOG_89_39
    template<>
    struct LogTableValueSize<89, 39, true>
    {
        typedef LogValuesTableQ89_39 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_89_39

    #if TINYMIND_USE_LOG_90_38
    template<>
    struct LogTableValueSize<90, 38, true>
    {
        typedef LogValuesTableQ90_38 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_90_38

    #if TINYMIND_USE_LOG_91_37
    template<>
    struct LogTableValueSize<91, 37, true>
    {
        typedef LogValuesTableQ91_37 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_91_37

    #if TINYMIND_USE_LOG_92_36
    template<>
    struct LogTableValueSize<92, 36, true>
    {
        typedef LogValuesTableQ92_36 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_92_36

    #if TINYMIND_USE_LOG_93_35
    template<>
    struct LogTableValueSize<93, 35, true>
    {
        typedef LogValuesTableQ93_35 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_93_35

    #if TINYMIND_USE_LOG_94_34
    template<>
    struct LogTableValueSize<94, 34, true>
    {
        typedef LogValuesTableQ94_34 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_94_34

    #if TINYMIND_USE_LOG_95_33
    template<>
    struct LogTableValueSize<95, 33, true>
    {
        typedef LogValuesTableQ95_33 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_95_33

    #if TINYMIND_USE_LOG_96_32
    template<>
    struct LogTableValueSize<96, 32, true>
    {
        typedef LogValuesTableQ96_32 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_96_32

    #if TINYMIND_USE_LOG_97_31
    template<>
    struct LogTableValueSize<97, 31, true>
    {
        typedef LogValuesTableQ97_31 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_97_31

    #if TINYMIND_USE_LOG_98_30
    template<>
    struct LogTableValueSize<98, 30, true>
    {
        typedef LogValuesTableQ98_30 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_98_30

    #if TINYMIND_USE_LOG_99_29
    template<>
    struct LogTableValueSize<99, 29, true>
    {
        typedef LogValuesTableQ99_29 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_99_29

    #if TINYMIND_USE_LOG_100_28
    template<>
    struct LogTableValueSize<100, 28, true>
    {
        typedef LogValuesTableQ100_28 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_100_28

    #if TINYMIND_USE_LOG_101_27
    template<>
    struct LogTableValueSize<101, 27, true>
    {
        typedef LogValuesTableQ101_27 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_101_27

    #if TINYMIND_USE_LOG_102_26
    template<>
    struct LogTableValueSize<102, 26, true>
    {
        typedef LogValuesTableQ102_26 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_102_26

    #if TINYMIND_USE_LOG_103_25
    template<>
    struct LogTableValueSize<103, 25, true>
    {
        typedef LogValuesTableQ103_25 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_103_25

    #if TINYMIND_USE_LOG_104_24
    template<>
    struct LogTableValueSize<104, 24, true>
    {
        typedef LogValuesTableQ104_24 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_104_24

    #if TINYMIND_USE_LOG_105_23
    template<>
    struct LogTableValueSize<105, 23, true>
    {
        typedef LogValuesTableQ105_23 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_105_23

    #if TINYMIND_USE_LOG_106_22
    template<>
    struct LogTableValueSize<106, 22, true>
    {
        typedef LogValuesTableQ106_22 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_106_22

    #if TINYMIND_USE_LOG_107_21
    template<>
    struct LogTableValueSize<107, 21, true>
    {
        typedef LogValuesTableQ107_21 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_107_21

    #if TINYMIND_USE_LOG_108_20
    template<>
    struct LogTableValueSize<108, 20, true>
    {
        typedef LogValuesTableQ108_20 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_108_20

    #if TINYMIND_USE_LOG_109_19
    template<>
    struct LogTableValueSize<109, 19, true>
    {
        typedef LogValuesTableQ109_19 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_109_19

    #if TINYMIND_USE_LOG_110_18
    template<>
    struct LogTableValueSize<110, 18, true>
    {
        typedef LogValuesTableQ110_18 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_110_18

    #if TINYMIND_USE_LOG_111_17
    template<>
    struct LogTableValueSize<111, 17, true>
    {
        typedef LogValuesTableQ111_17 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_111_17

    #if TINYMIND_USE_LOG_112_16
    template<>
    struct LogTableValueSize<112, 16, true>
    {
        typedef LogValuesTableQ112_16 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_112_16

    #if TINYMIND_USE_LOG_113_15
    template<>
    struct LogTableValueSize<113, 15, true>
    {
        typedef LogValuesTableQ113_15 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_113_15

    #if TINYMIND_USE_LOG_114_14
    template<>
    struct LogTableValueSize<114, 14, true>
    {
        typedef LogValuesTableQ114_14 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_114_14

    #if TINYMIND_USE_LOG_115_13
    template<>
    struct LogTableValueSize<115, 13, true>
    {
        typedef LogValuesTableQ115_13 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_115_13

    #if TINYMIND_USE_LOG_116_12
    template<>
    struct LogTableValueSize<116, 12, true>
    {
        typedef LogValuesTableQ116_12 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_116_12

    #if TINYMIND_USE_LOG_117_11
    template<>
    struct LogTableValueSize<117, 11, true>
    {
        typedef LogValuesTableQ117_11 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_117_11

    #if TINYMIND_USE_LOG_118_10
    template<>
    struct LogTableValueSize<118, 10, true>
    {
        typedef LogValuesTableQ118_10 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_118_10

    #if TINYMIND_USE_LOG_119_9
    template<>
    struct LogTableValueSize<119, 9, true>
    {
        typedef LogValuesTableQ119_9 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_119_9

    #if TINYMIND_USE_LOG_120_8
    template<>
    struct LogTableValueSize<120, 8, true>
    {
        typedef LogValuesTableQ120_8 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_120_8

    #if TINYMIND_USE_LOG_121_7
    template<>
    struct LogTableValueSize<121, 7, true>
    {
        typedef LogValuesTableQ121_7 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_121_7

    #if TINYMIND_USE_LOG_122_6
    template<>
    struct LogTableValueSize<122, 6, true>
    {
        typedef LogValuesTableQ122_6 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_122_6

    #if TINYMIND_USE_LOG_123_5
    template<>
    struct LogTableValueSize<123, 5, true>
    {
        typedef LogValuesTableQ123_5 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_123_5

    #if TINYMIND_USE_LOG_124_4
    template<>
    struct LogTableValueSize<124, 4, true>
    {
        typedef LogValuesTableQ124_4 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_124_4

    #if TINYMIND_USE_LOG_125_3
    template<>
    struct LogTableValueSize<125, 3, true>
    {
        typedef LogValuesTableQ125_3 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_125_3

    #if TINYMIND_USE_LOG_126_2
    template<>
    struct LogTableValueSize<126, 2, true>
    {
        typedef LogValuesTableQ126_2 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_126_2

    #if TINYMIND_USE_LOG_127_1
    template<>
    struct LogTableValueSize<127, 1, true>
    {
        typedef LogValuesTableQ127_1 LogTableType;
    };
    #endif // TINYMIND_USE_LOG_127_1

    template<unsigned FixedBits,unsigned FracBits, bool IsSigned>
    struct LogValuesTableSelector
    {
        typedef typename LogTableValueSize<FixedBits, FracBits, IsSigned>::LogTableType LogTableType;
    };
}
