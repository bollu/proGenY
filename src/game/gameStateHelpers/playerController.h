#pragma once
#include "../../core/controlFlow/EventManager.h"
#include "../../core/Rendering/viewProcess.h"

class gunsManager;
class playerEventHandler;
class playerCreator;
struct playerHandlerData;
class ObjectManager;
class Object;
class objectFactory;

class playerController{
public:
	playerController(EventManager *_eventManager, ObjectManager *objectManager, objectFactory *factory, 
		viewProcess *viewProc);

	void createPlayer(vector2 levelDim, 
		vector2 initPos, playerCreator *creator, playerHandlerData playerData);	
	
	
	void Update(float dt);

	Object *getPlayer();

private:

	EventManager *_eventManager;
	ObjectManager *_objectManager;
	objectFactory *_objectFactory;
	viewProcess *viewProc;
	
	Object *player;
	gunsManager *gunsMgr;
	playerEventHandler *playerHandler;

	vector2 levelDim;

	void _createPlayer(vector2 initPos, playerCreator *creator);
	void _createGunsManager(Object *player);
	void _createPlayerEventHandler(playerHandlerData &playerData);
};
