#pragma once
#include "ObjectProcessor.h"
#include "../../IO/logObject.h"

#include "../../Rendering/viewProcess.h"
#include "../../World/worldProcess.h"

#include "../../controlFlow/processMgr.h"
#include "../../IO/Settings.h"
#include "../../controlFlow/EventManager.h"
#include "../../World/objContactListener.h"

//ALYWAS CREATE THE FIXTURE DEF's SHAPE ON THE STACK
/*!Stores data used by the PhyProcessor

stores all essential information that is needed to 
run the physics of Objects
*/
struct PhyData{
	/*!the definition of the body */
	//b2BodyDef bodyDef;
	/*!a list of definitions of all fixtures owned by the body*/
	//std::vector<b2FixtureDef> fixtureDef;

	/*!a pointer to the body
	--don't modify this without knowing what you're doing--
	*/
	b2Body *body;
	/* a list of pointers to the fixtures owned by the body*/
	//std::vector<b2Fixture*>fixtures;

	/*!whether the velocity should be clamped between -maxVel and  maxVel or not*/
	bool velClamped;
	/*!the velocity to which the body's velocity should be clamped if 
	PhyData::velClamped is true*/
	vector2 maxVel;
	
	/*!the name of the collisionType of the body.
	Used while filtering collisions
	*/
	const Hash* collisionType;
	std::vector<CollisionHandler> collisionHandlers;

private:
	friend class PhyProcessor;
	//must be constructed by calling createPhyData()
	PhyData(){};
	//friend class objContactListener;
	// void addCollision(CollisionData &collision);
	// void removeCollision(Object *obj);
};




/*!Processes physics data of Object classes.

This ObjectProcessor acts as a proxy between box2d and the game.
it is responsible for creating box2d bodies and shapes when an 
object is first added. It is also responsible for updating the
position and angle of the body every frame. It must also destroy
all box2d related data when the Object is killed or removed
*/
class PhyProcessor : public ObjectProcessor{
private:
	viewProcess *view;
	static worldProcess *world;
	objContactListener contactListener;

public:
	PhyProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager);
	static PhyData createPhyData(b2BodyDef *bodyDef, b2FixtureDef fixtures[], unsigned int numFixtures = 1);

protected:
	void _onObjectAdd(Object *obj);

	void _onObjectActivate(Object *obj);
	void _onObjectDeactivate(Object *obj);

	void _preProcess();
	void _Process(Object *obj, float dt);
	void _onObjectDeath(Object *obj);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("PhyData");
	};

};