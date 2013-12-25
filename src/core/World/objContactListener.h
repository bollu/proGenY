#pragma once
#include "../../include/Box2D/Box2D.h"
#include "../math/vector.h"
#include "../IO/Hash.h"

struct PhyData;
class Object;

/*!Stores collision related data*/
struct CollisionData{
	/*!the physicsData of the *other* object that this object
	has collided with */ 
	PhyData *otherPhy;
	/*!the *other* object that this object has collided with*/
	Object *otherObj;	


	/*!the physicsData of *this* object */
	PhyData *myPhy;
	Object *me;

	b2Contact* contact;

	//const Hash *getCollidedObjectCollision();
};


typedef void (*CollisionCallback) (CollisionData &collision, void *data);
struct CollisionHandler{
	//!the collision type of the object you want to hit.
	//if this is ***NULL**, then the collisionHandler will be called for all collisions
	const Hash *otherCollision = (const Hash*)(0xDEADBEEF);
	CollisionCallback onBegin = NULL;
	CollisionCallback onEnd = NULL;

	void *data = NULL;
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
public:
	objContactListener(){};
	~objContactListener(){};

	void BeginContact(b2Contact* contact);
	void EndContact(b2Contact* contact);

private:
	void _extractObjects(b2Contact *contact, Object **a, Object **b);

	/*CollisionData _fillCollisionData(b2Contact *contact,
  Object *me, Object *other, PhyData *myPhy, PhyData *otherPhy);
	*/

	//HACK! should be CollisionData::Type, but this creates a
	//cyclic dependency :(
	//void _handleCollision(CollisionData::Type type, b2Contact *contact);
    bool _shouldHandleCollision(CollisionHandler *handler, PhyData *otherPhy);
    void _handleCollision(bool beginHandler, b2Contact *contact);
    void _execHandlers(bool beginHandler, b2Contact *contact, Object *me, PhyData *myData, Object *other, PhyData *otherData);

   
};
