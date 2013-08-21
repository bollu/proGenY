#pragma once
#include "../objectProcessor.h"
#include "../../util/logObject.h"

#include "../Process/viewProcess.h"
#include "../Process/worldProcess.h"

#include "../Process/processMgr.h"
#include "../Settings.h"
#include "../Messaging/eventMgr.h"


#include "objContactListener.h"

class phyProp;

//ALYWAS CREATE THE FIXTURE DEF's SHAPE ON THE STACK

/*!Stores data used by the phyProcessor

   stores all essential information that is needed to
   run the physics of Objects
 */
class phyProp : public baseProperty{
public:
	/*!the definition of the body */
	b2BodyDef		    bodyDef;
	/*!a list of definitions of all fixtures owned by the body*/
	std::vector< b2FixtureDef > fixtureDef;
	/*!a pointer to the body
	   --don't modify this without knowing what you're doing--*/
	b2Body			  *body;
	/* a list of pointers to the fixtures owned by the body*/
	std::vector< b2Fixture * > fixtures;
	/*!whether the velocity should be clamped between -maxVel and  maxVel or not*/
	bool  velClamped;
	/*!the velocity to which the body's velocity should be clamped if
	   phyProp::velClamped is true*/
	vector2 maxVel;

	/*!the name of the collisionType of the body.
	   Used while filtering collisions
	 */
	const Hash * collisionType;

	/*a list of all collisions that took place this frame.
	   NOTE- this list is cleared every frame
	 */
	std::vector< collisionData > collisions;

	~phyProp(){
		util::infoLog<<"\n phyData is destroyed";
	}

private:
	friend class objContactListener;
	void addCollision ( collisionData &collision );
	void removeCollision ( Object *obj );
};


/*!Processes physics data of Object classes.

   This objectProcessor acts as a proxy between box2d and the game.
   it is responsible for creating box2d bodies and shapes when an
   object is first added. It is also responsible for updating the
   position and angle of the body every frame. It must also destroy
   all box2d related data when the Object is killed or removed
 */
class phyProcessor : public objectProcessor
{
private:
	b2World *world;
	viewProcess *view;
	objContactListener contactListener;


public:
	phyProcessor ( processMgr &processManager, Settings &settings, eventMgr &_eventManager );
	void onObjectAdd ( Object *obj );
	void preProcess ();
	void Process ( float dt );
	void postProcess ();
	void onObjectRemove ( Object *obj );
};