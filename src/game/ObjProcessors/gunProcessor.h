#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

#include "../../core/Process/viewProcess.h"
#include "../../core/Process/worldProcess.h"


#include "../factory/bulletCreator.h"

#include "../../util/mathUtil.h"


struct gunData{
private:

	friend class gunProcessor;

	int totalClipSize;
	//ticks down from currentAmmo to zero
	int currentClipSize;

	//time needed to reload the entire ammo
	int totalClipCooldown;
	//ticks down from totalClipCooldown to 0
	int currentClipCooldown;

	bool clipOnCooldown;

	//time between 2 different shots
	int totalShotCooldown;
	//ticks down from totalCooldown to 0
	int currrentShotCooldown;

	bool shotOnCooldown;

	util::Angle facing;

	bool firing;

	void _fireShot();
public:
	gunData(){
		this->shotOnCooldown = false;
		this->clipOnCooldown = false;

		this->totalClipSize = this->currentClipSize = 0;
		this->totalClipCooldown = this->currentClipCooldown = 0;
		this->totalShotCooldown = this->currrentShotCooldown = 0;
		
		this->firing = false;
	}
	bulletCreator *bullet;

	void setClipSize(int totalClipSize){
		this->totalClipSize = totalClipSize;
	}

	void setClipCooldown(int totalClipCooldown){
		this->totalClipCooldown = totalClipCooldown;
	}

	void setShotCooldown(int totalShotCooldown){
		this->totalShotCooldown = totalShotCooldown;
	}

	void setFacing(util::Angle facing){
		this->facing = facing;
	}

	

	void Tick();

	bool shouldFire(){
		return this->firing;
	}
};


class gunProcessor : public objectProcessor{
public:
	gunProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){};
	
	void Process(float dt);
	void postProcess();
};