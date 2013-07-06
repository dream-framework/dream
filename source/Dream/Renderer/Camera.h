//
//  Renderer/Camera.h
//  This file is part of the "Dream" project, and is released under the MIT license.
//
//  Created by Samuel Williams on 9/06/06.
//  Copyright (c) 2006 Samuel Williams. All rights reserved.
//
//

#ifndef _DREAM_RENDERER_CAMERA_H
#define _DREAM_RENDERER_CAMERA_H

#include "Renderer.h"

#include <Euclid/Numerics/Matrix.h>
#include <Euclid/Numerics/Vector.h>

namespace Dream
{
	namespace Renderer
	{
		using Euclid::Numerics::Mat44;
		using Euclid::Numerics::Vec3;
		

		class ICamera : implements IObject {
		public:
			virtual ~ICamera ();

			virtual Mat44 view_matrix() const abstract;
		};

		class Camera : implements ICamera {
		protected:
			Mat44 _view_matrix;

		public:
			Camera(const Mat44 & view_matrix);
			virtual ~Camera ();

			virtual Mat44 view_matrix() const;
			Mat44 & view_matrix();
		};
	}
}

#endif
