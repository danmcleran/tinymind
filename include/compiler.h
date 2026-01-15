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
 *   TINYMIND_DISABLE_WARNING_PUSH
 *   TINYMIND_DISABLE_WARNING("-Wwhatever")     // for GCC/Clang
 *   TINYMIND_DISABLE_WARNING(4996)              // for MSVC (numeric code)
 *   TINYMIND_DISABLE_WARNING_POP
 * 
 * Example: disable dangling-reference on GCC/Clang (no-op on MSVC)
 *   TINYMIND_DISABLE_WARNING_PUSH
 *   TINYMIND_DISABLE_WARNING("-Wdangling-reference")
 *   TINYMIND_DISABLE_WARNING_POP
 *
 * Notes:
 *  - For GCC/Clang pass the warning name as a quoted string (e.g. "-Wdeprecated-declarations").
 *  - For MSVC pass the numeric warning code (e.g. 4100, 4996).
 */

#if defined(_MSC_VER)
	#define TINYMIND_PRAGMA(x) __pragma(x)
	#define TINYMIND_DISABLE_WARNING_PUSH TINYMIND_PRAGMA(warning(push))
	#define TINYMIND_DISABLE_WARNING_POP  TINYMIND_PRAGMA(warning(pop))
	#define TINYMIND_DISABLE_WARNING_MSVC(code) TINYMIND_PRAGMA(warning(disable: code))
	/* GCC/Clang style no-op on MSVC */
	#define TINYMIND_DISABLE_WARNING_GCC_CLANG(warning)
	/* Generic helper maps to MSVC form on MSVC */
	#define TINYMIND_DISABLE_WARNING(w) TINYMIND_DISABLE_WARNING_MSVC(w)
#elif defined(__clang__) || defined(__GNUC__)
	#define TINYMIND_PRAGMA(x) _Pragma(#x)
	#define TINYMIND_DISABLE_WARNING_PUSH TINYMIND_PRAGMA(GCC diagnostic push)
	#define TINYMIND_DISABLE_WARNING_POP  TINYMIND_PRAGMA(GCC diagnostic pop)
	#define TINYMIND_DISABLE_WARNING_GCC_CLANG(warning) TINYMIND_PRAGMA(GCC diagnostic ignored warning)
	/* MSVC form no-op on GCC/Clang */
	#define TINYMIND_DISABLE_WARNING_MSVC(code)
	/* Generic helper maps to GCC/Clang form on GCC/Clang */
	#define TINYMIND_DISABLE_WARNING(w) TINYMIND_DISABLE_WARNING_GCC_CLANG(w)
#else
	/* Unknown compiler: define no-ops so code using these macros stays portable */
	#define TINYMIND_PRAGMA(x)
	#define TINYMIND_DISABLE_WARNING_PUSH
	#define TINYMIND_DISABLE_WARNING_POP
	#define TINYMIND_DISABLE_WARNING_MSVC(code)
	#define TINYMIND_DISABLE_WARNING_GCC_CLANG(warning)
	#define TINYMIND_DISABLE_WARNING(w)
#endif