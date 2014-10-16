//
//  Assertion.h
//  This file is part of the "Dream" project, and is released under the MIT license.
//
//  Created by Samuel Williams on 26/10/08.
//  Copyright (c) 2008 Samuel Williams. All rights reserved.
//
//

#ifndef _DREAM_ASSERTION_H
#define _DREAM_ASSERTION_H

#include <string>
#include <exception>

#ifdef DREAM_RELEASE
	#define DREAM_ASSERT(e)
#else
	/// Similar to assert, however will throw an AssertionError on failure.
	#define DREAM_ASSERT(e) ::Dream::AssertionError::assert_handler(e, # e, __FILE__, __LINE__)
#endif

namespace Dream
{
	/**
	 An exception class that indicates a failed assertion.

	 Use the ensure() macro in your code:

	 @code
	 ensure(my_condition)
	 @endcode
	 */
	class AssertionError : public std::exception {
		const char * _expression;
		const char * _file;
		unsigned _line;

		std::string _what;
	public:
		AssertionError (const char * expression, const char * file, unsigned line) throw ();
		virtual ~AssertionError () throw ();

		/// A descriptive string relating to the assertion that failed.
		virtual const char * what () const throw ();

		/// The DREAM_ASSERT() macro calls this function to handle throwing the actual exception.
		static void assert_handler (bool condition, const char * expression, const char * file, unsigned line);
	};
}

#endif
