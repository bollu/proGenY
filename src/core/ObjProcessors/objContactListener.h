#pragma once
#include "../../include/Box2D/Box2D.h"


class phyData;
class Object;

/*!converts box2d collisions to high-level game collisions.

the objContactListener binds to box2d and listens to all collisions
if a collision meets the criteria needed for it to be considered a 
"game-level" event, the objContactListener converts the low level
box2d event to a high level game collision event. 

This class is responsible for filling phyData::collisions

\sa phyData
\sa phyProcessor
*/
class objContactListener : public b2ContactListener{
public:
	objContactListener(){};
	~objContactListener(){};

	void BeginContact(b2Contact* contact);
    void EndContact(b2Contact* contact);



	void _extractPhyData(b2Contact *contact, Object **a, Object **b);
};