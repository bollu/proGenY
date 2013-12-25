#pragma once
#include "../../core/controlFlow/EventManager.h"
#include "../../core/Rendering/viewProcess.h"

class gunsManager;
class playerEventHandler;
class playerCreator;
struct playerHandlerData;
class ObjectManager;
class Object;

class playerController{
public:
	playerController(EventManager *_eventManager, ObjectManager *objectManager, viewProcess *viewProc);

	void createPlayer(vector2 levelDim, vector2 initPos, playerHandlerData playerData);
	void Update(float dt);

private:

	EventManager *eventManager_;
	ObjectManager *objectManager_;
	viewProcess *viewProc_;
	
	Object *player_;
	gunsManager *gunsMgr_;
	playerEventHandler *playerHandler_;


	void _createPlayer(vector2 initPos, playerCreator *creator);
	void _createGunsManager(Object *player);
	void _createPlayerEventHandler(playerHandlerData &playerData);
};
