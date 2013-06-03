#pragma once
#include "../../include/Box2D/Box2D.h"


class phyData;
class Object;

class objContactListener : public b2ContactListener{
public:
	objContactListener(){};
	~objContactListener(){};

	void BeginContact(b2Contact* contact);
    void EndContact(b2Contact* contact);



	void _extractPhyData(b2Contact *contact, Object **a, Object **b);
};