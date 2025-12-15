/**
* Copyright (c) 2025 Dan McLeran
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

/*
 * Cross-compiler macros to push/pop diagnostics and disable warnings.
 *
 * Usage:
 *   DISABLE_WARNING_PUSH
 *   DISABLE_WARNING("-Wwhatever")     // for GCC/Clang
 *   DISABLE_WARNING(4996)              // for MSVC (numeric code)
 *   DISABLE_WARNING_POP
 * 
 * Example: disable dangling-reference on GCC/Clang (no-op on MSVC)
 *   DISABLE_WARNING_PUSH
 *   DISABLE_WARNING("-Wdangling-reference")
 *   DISABLE_WARNING_POP
 *
 * Notes:
 *  - For GCC/Clang pass the warning name as a quoted string (e.g. "-Wdeprecated-declarations").
 *  - For MSVC pass the numeric warning code (e.g. 4100, 4996).
 */

#if defined(_MSC_VER)
	#define TM_PRAGMA(x) __pragma(x)
	#define DISABLE_WARNING_PUSH TM_PRAGMA(warning(push))
	#define DISABLE_WARNING_POP  TM_PRAGMA(warning(pop))
	#define DISABLE_WARNING_MSVC(code) TM_PRAGMA(warning(disable: code))
	/* GCC/Clang style no-op on MSVC */
	#define DISABLE_WARNING_GCC_CLANG(warning)
	/* Generic helper maps to MSVC form on MSVC */
	#define DISABLE_WARNING(w) DISABLE_WARNING_MSVC(w)
#elif defined(__clang__) || defined(__GNUC__)
	#define TM_PRAGMA(x) _Pragma(#x)
	#define DISABLE_WARNING_PUSH TM_PRAGMA(GCC diagnostic push)
	#define DISABLE_WARNING_POP  TM_PRAGMA(GCC diagnostic pop)
	#define DISABLE_WARNING_GCC_CLANG(warning) TM_PRAGMA(GCC diagnostic ignored warning)
	/* MSVC form no-op on GCC/Clang */
	#define DISABLE_WARNING_MSVC(code)
	/* Generic helper maps to GCC/Clang form on GCC/Clang */
	#define DISABLE_WARNING(w) DISABLE_WARNING_GCC_CLANG(w)
#else
	/* Unknown compiler: define no-ops so code using these macros stays portable */
	#define TM_PRAGMA(x)
	#define DISABLE_WARNING_PUSH
	#define DISABLE_WARNING_POP
	#define DISABLE_WARNING_MSVC(code)
	#define DISABLE_WARNING_GCC_CLANG(warning)
	#define DISABLE_WARNING(w)
#endif