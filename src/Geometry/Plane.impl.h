//
//  Geometry/Plane.impl.h
//  This file is part of the "Dream" project, and is released under the MIT license.
//
//  Created by Samuel Williams on 2/12/08.
//  Copyright (c) 2008 Samuel Williams. All rights reserved.
//
//

#ifndef _DREAM_GEOMETRY_PLANE_H
#error This header should not be included manually. Include Plane.h instead.
#endif

namespace Dream
{
	namespace Geometry
	{
		template <unsigned D, typename NumericT>
		bool Plane<D, NumericT>::is_parallel (const Plane<D, NumericT> & other) const
		{
			return _normal == other._normal || _normal == (- other._normal);
		}

		template <unsigned D, typename NumericT>
		bool Plane<D, NumericT>::intersects_with (const Plane<D, NumericT> & other, Line<3, NumericT> & line) const
		{
			/* Planes are parallel? */
			if (other.normal() == _normal)
				return false;
			
			VectorT u = _normal.cross(other._normal).normalize();
			
			line.set_direction(u);
			line.set_point(- ((_normal*other._distance) - (other._normal*_distance)).cross(u) / u.length2());
			
			return true;	
		}
		
		template <unsigned D, typename NumericT>
		bool Plane<D, NumericT>::intersects_with (const Line<3, NumericT> & line, VectorT & at) const
		{
			NumericT d = _normal.dot(line.direction());
			
			/* Line and Plane are parallel? */
			if (d == 0.0) return false;
			
            // This minus sign may need to be inside the (-_normal)
			NumericT r = -(_normal.dot(line.point()) - _distance);
			NumericT t = r / d;
			
			at = line.point() + line.direction() * t;
			
			return true;
		}
		
		template <unsigned D, typename NumericT>
		IntersectionResult Plane<D, NumericT>::intersects_with (const Sphere<D, NumericT> & sphere) const
		{
			NumericT d = distance_to_point(sphere.center());
			
			if (d > sphere.radius())
				return NO_INTERSECTION;
			else if (Number<NumericT>::equivalent(d, sphere.radius()))
				return EDGES_INTERSECT;
			else
				return SHAPES_INTERSECT;
		}

		template <unsigned D, typename NumericT>
		std::ostream &operator<<(std::ostream &out, const Plane<D, NumericT> & p) {
			return out << "norm: " << p.normal() << " d: " << p.distance();
		}		
		
	}
}
