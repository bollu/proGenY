
#include "AABB.h"


AABB::AABB(vector2 _halfDim, vector2 _center) : center(_center), halfDim(_halfDim){
};
AABB::AABB(vector2 _halfDim) : center(0,0) , halfDim(_halfDim) {
};

AABB AABB::Endpoints(vector2 bottomLeft, vector2 topRight){
	vector2 dim = topRight - bottomLeft;
	return AABB(dim * 0.5, (bottomLeft + topRight) * 0.5); 
}

vector2 AABB::getDim() const{
	return this->halfDim * 2;
};

vector2 AABB::getHalfDim() const{
	return this->halfDim;
};

vector2 AABB::getCenter() const{
	return this->center;
};

vector2 AABB::getTopLeft() const{
	return this->center + vector2(-halfDim.x, halfDim.y);
};

vector2 AABB::getTopRight() const{
	return this->center + this->halfDim;
};
vector2 AABB::getBottomLeft() const{
	return this->center - this->halfDim;
};

vector2 AABB::getBottomRight() const{
	return this->center + vector2(halfDim.x, -halfDim.y);
};

bool AABB::Intersects(const AABB &other) const{
	vector2 dist = other.center - this->center;
	dist.x = abs(dist.x);
	dist.y = abs(dist.y);

	vector2 maxDist = this->halfDim + other.halfDim;

	return (dist.x <= maxDist.x && dist.y <= maxDist.y);
};

bool AABB::Contains(const vector2 &pt) const{
	vector2 dist = pt - center;

	if(abs(dist.x) < halfDim.x && abs(dist.y) < halfDim.y){
		return true;
	}

	return false;
};

bool AABB::liesOn(const vector2 &pt) const{
	vector2 dist = pt - center;

	if(abs(dist.x) == halfDim.x && abs(dist.y) <= halfDim.y){
		return true;
	}
	else if(abs(dist.y) == halfDim.y && abs(dist.x) <= halfDim.x){
		return true;
	}

	return false;
};
