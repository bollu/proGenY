#pragma once
#include "../../include/Box2D/Box2D.h"
#include "../math/vector.h"
#include "../IO/Hash.h"

class PhyData;
class Object;

struct collisionData;


/*!Stores collision related data*/
struct collisionData{
	/*!the type of collision*/
	enum Type{
		/*!the collision has just begun*/
		onBegin,
		/*!the collision has just ended*/
		onEnd,
	} type;
	
	/*!the physicsData of the *other* object that this object
	has collided with */ 
	PhyData *otherPhy;
	/*!the physicsData of *this* object */
	PhyData *myPhy;
	/*!the *other* object that this object has collided with*/
	Object *otherObj;	

	vector2 normal;

	vector2 myApproachVel;

	const Hash *getCollidedObjectCollision();
};




/*!converts box2d collisions to high-level game collisions.

the objContactListener binds to box2d and listens to all collisions
if a collision meets the criteria needed for it to be considered a 
"game-level" event, the objContactListener converts the low level
box2d event to a high level game collision event. 

This class is responsible for filling PhyData::collisions

\sa PhyData
\sa PhyProcessor
*/
class objContactListener : public b2ContactListener{
private:
	void _extractPhyData(b2Contact *contact, Object **a, Object **b);

	collisionData _fillCollisionData(b2Contact *contact,
  Object *me, Object *other, PhyData *myPhy, PhyData *otherPhy);

	//HACK! should be collisionData::Type, but this creates a
	//cyclic dependency :(
	void _handleCollision(collisionData::Type type, b2Contact *contact);
public:
	objContactListener(){};
	~objContactListener(){};

	void BeginContact(b2Contact* contact);
    void EndContact(b2Contact* contact);



	
};
