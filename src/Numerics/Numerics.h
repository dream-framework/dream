//
//  Numerics/Numerics.h
//  This file is part of the "Dream" project, and is released under the MIT license.
//
//  Created by Samuel Williams on 13/05/06.
//  Copyright (c) 2008 Samuel Williams. All rights reserved.
//
//

#ifndef _DREAM_NUMERICS_NUMERICS_H
#define _DREAM_NUMERICS_NUMERICS_H

#include "../Framework.h"

#include <cmath>

namespace Dream
{
	/** Mathematics and functionality directly associated with numbers.
	 */
	namespace Numerics
	{
		typedef float single;

		/// Default floating point type
		typedef single RealT;

		/// Zero type. Used in some Numerics constructors.
		enum Zero
		{
			ZERO = 0
		};
		
		/// Identity type. Used in some Numerics constructors.
		enum Identity
		{
			IDENTITY = 1
		};

		/// 4D Vector indices, for use with Vector
		enum
		{
			X = 0, Y = 1, Z = 2, W = 3
		};

		/// 3D size indicies, for use with Vector
		enum
		{
			WIDTH = 0, HEIGHT = 1, DEPTH = 2
		};

		/// 10 degree rotation
		const double R10 = M_PI_2 / (double) 9.0;
		/// 30 degree rotation
		const double R30 = M_PI_2 / (double) 3.0;
		/// 45 degree rotation
		const double R45 = M_PI_4;
		/// 60 degree rotation
		const double R60 = R30*2;
		/// 90 degree rotation
		const double R90 = M_PI_2;
		/// 180 degree rotation
		const double R180 = M_PI;
		/// 270 degree rotation
		const double R270 = R90*3;
		/// 360 degree rotation
		const double R360 = R180*2;

		/// Radians to degrees multiplier
		const double R2D = (180.0 / M_PI);
		const double D2R = (M_PI / 180.0);
	
		/// If the supplied value is a power of two, it is returned, otherwise the next highest power of 2 is calculated and returned.
		uint32_t next_highest_power_of2 (uint32_t);

		/// Checks if an unsigned int is a power of 2.
		bool is_power_of2 (uint32_t);

		enum
		{
			DEFAULT_ULPS = 100
		};

		/// Checks the equivalence of two doubles.
		/// As double is not a precise type, ulps specifies the exact distance between permutations that is acceptable. Be aware, that as numbers get closer to
		/// zero, the distance between fixed permutations gets smaller.
		bool equal_within_tolerance (const double & a, const double & b, const unsigned & ulps = DEFAULT_ULPS);

		/// Checks the equivalence of two floats.
		/// As float is not a precise type, ulps specifies the exact distance between permutations that is acceptable. Be aware, that as numbers get closer to
		/// zero, the distance between fixed permutations gets smaller.
		bool equal_within_tolerance (const float & a, const float & b, const unsigned & ulps = DEFAULT_ULPS);

		/// Equivalence of floating point numbers.
		/// This function is not "accurate" in the sense that it considers only about 5 decimal places at best. But it provides a robust and reliable method to
		/// compare floating point numbers in the sense that the libm trig functions are not very reliable either.
		bool equivalent (const float & a, const float & b);

		/// Equivalence of floating point numbers.
		bool equivalent (const double & a, const double & b);

		/// Helper to get floating point type from a fixed point type
		template <typename t>
		struct RealType
		{
			typedef float RealT;
		};

		/// Helper to get floating point type from a fixed point type
		template <>
		struct RealType<double>
		{
			typedef double RealT;
		};
	}
}

#endif
