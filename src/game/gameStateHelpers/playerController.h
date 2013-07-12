#pragma once
#include "../../core/Messaging/eventMgr.h"
#include "../../core/Process/viewProcess.h"

class gunsManager;
class playerEventHandler;
class playerCreator;
class playerHandlerData;
class objectMgr;
class Object;

class playerController{
public:
	playerController(eventMgr *_eventManager, objectMgr *objectManager, viewProcess *viewProc);

	void createPlayer(vector2 levelDim, 
		vector2 initPos, playerCreator *creator,playerHandlerData playerData);	
	void addGun(Object *gun, bool currentGun=false);

	void Update(float dt);

	Object *getPlayer();

private:

	eventMgr *_eventManager;
	objectMgr *_objectManager;
	viewProcess *viewProc;
	
	Object *player;
	gunsManager *gunsMgr;
	playerEventHandler *playerHandler;

	vector2 levelDim;

	void _createPlayer(vector2 initPos, playerCreator *creator);
	void _createGunsManager(Object *player);
	void _createPlayerEventHandler(playerHandlerData &playerData);
};