
#include "playerController.h"
#include "gunsManager.h"
#include "playerEventHandler.h"

#include "../../core/componentSys/Object.h"
#include "../../core/Rendering/renderUtil.h"

#include "../../core/controlFlow/EventManager.h"
#include "../../core/componentSys/ObjectManager.h"

#include "../factory/objectFactories.h"


playerController::playerController(EventManager *eventManager, ObjectManager *objectManager, viewProcess *viewProc){
	
	this->eventManager_ = eventManager;
	this->objectManager_ = objectManager;
	this->viewProc_ = viewProc;

	this->gunsMgr_ = NULL;
	this->playerHandler_ = NULL;

	
};

void playerController::createPlayer(vector2 levelDim, vector2 initPos, playerHandlerData playerData){


	//camera data--------------------------------------
	ObjectFactories::PlayerFactoryInfo playerFactoryInfo;
	playerFactoryInfo.viewProc = this->viewProc_;

	CameraData &cameraData = playerFactoryInfo.cameraData;
	cameraData.enabled = true;
	cameraData.maxCoord = (viewProc_->game2ViewCoord(levelDim));
	cameraData.maxMoveAmt = vector2(30, 60);
	cameraData.boxHalfW = 360;
	cameraData.boxHalfH = 200;

	//player created--------------------------------------------
	playerFactoryInfo.pos = initPos;

	player_ = ObjectFactories::CreatePlayer(playerFactoryInfo);
	IO::infoLog<<"player created";

	//create guns
	_createGunsManager(player_);

	//create playerEventHandler
	playerData.player = player_;
	_createPlayerEventHandler(playerData);
	IO::infoLog<<"player event handler created";

	//add player
	objectManager_->addObject(player_);
	IO::infoLog<<"added to object manager";

};


void playerController::Update(float dt)
{
	assert(this->playerHandler_ != NULL);
	playerHandler_->Update();
};

Object *playerController::getPlayer(){
	assert(this->player_ != NULL);
	return this->player_;
};


void playerController::_createGunsManager(Object *player){
 	gunsMgr_ = new gunsManager(*this->eventManager_, *this->objectManager_, viewProc_, player);
};

void playerController::_createPlayerEventHandler(playerHandlerData &playerData)
{
	this->playerHandler_ = new playerEventHandler(this->eventManager_, playerData); 

};
