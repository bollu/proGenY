#pragma once
#include "vector.h"




class AABB{
public:
	AABB(vector2 halfDim, vector2 center);
	AABB(vector2 halfDim);

	static AABB Endpoints(vector2 bottomLeft, vector2 topRight);

	vector2 getDim() const;
	vector2 getHalfDim() const;
	vector2 getCenter() const;

	vector2 getTopLeft() const;
	vector2 getTopRight() const;
	vector2 getBottomLeft() const;
	vector2 getBottomRight() const;

	bool Intersects(const AABB &other) const;
	/*returns true if the point is *inside* the AABB */
	bool Contains(const vector2 &pt) const;
	/*!returns true if the point is *on* the AABB */
	bool liesOn(const vector2 &pt) const;

private:
	vector2 center;
	vector2 halfDim;
};
