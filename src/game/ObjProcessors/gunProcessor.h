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



struct gunData{
private:
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


public:

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

	void fireShot();

	void Tick();
};


class gunProcessor : public objProcessor{
public:
	void Process(float dt);
	void postProcess();
};