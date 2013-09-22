#pragma once
#include "../AABB.h"


enum quadtreeDir{
	topLeft = 0,
	topRight,
	bottomLeft,
	bottomRight
};

class quadtreeLeaf<T>{
public:
	quadtreeLeaf(AABB boundingBox);

	void Rebuild();
	std::list<T*> &getDepartedObjects();

	std::list<T*> getContainedObjects();
private:
	std::list<T*> objects;
};


class quadtreeNode<T>{
public:
	quadtreeNode(AABB boundingBox, int myDepth);
	void Rebuild();

	std::list<T*> &getDepartedObjects();
	void enterObject(T* object);

	std::list<T*> getContainedObjects();
private:
	AABB boundingBox;
	quadtreeNode<T>[4] nodes;
};


class Quadtree<T>{
public:
	Quadtree(vector2 totalDim, int totalDepth);

	void Rebuild();
	void AABBQuery(AABB box);
	void circularQuery(vector2 pos, float radius);
private:
	AABB boundingBox;
	quadtreeNode<T>[4] nodes;

};
