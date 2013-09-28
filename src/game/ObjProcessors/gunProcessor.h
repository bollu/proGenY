#pragma once
#include "../../core/componentSys/processor/objectProcessor.h"
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/eventMgr.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"


#include "../factory/bulletCreator.h"

#include "../../core/math/mathUtil.h"


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
	vector2 bulletPos;
	float bulletRadius;
	float buletVel;

	bool firing;

	bulletData bullet;

	void _Cooldown();
	void _Tick();
	bool _shouldFire(){
		return this->firing;
	}

public:
	bulletCreator *creator;
	
	gunData(){
		this->shotOnCooldown = false;
		this->clipOnCooldown = false;

		this->totalClipSize = this->currentClipSize = 0;
		this->totalClipCooldown = this->currentClipCooldown = 0;
		this->totalShotCooldown = this->currrentShotCooldown = 0;
		this->firing = false;

		this->creator = NULL;
	}
	
	//functions to be called during initialization
	void setClipSize(int totalClipSize){
		this->totalClipSize = totalClipSize;
		this->currentClipSize = this->totalClipSize;
		assert(this->totalClipSize > 0);
	}

	void setClipCooldown(int totalClipCooldown){
		this->totalClipCooldown = totalClipCooldown;
		assert(this->totalClipCooldown >= 0);
	}

	void setShotCooldown(int totalShotCooldown){
		this->totalShotCooldown = totalShotCooldown;
		assert(this->totalShotCooldown >= 0);
	}

	
	void setBulletRadius(float radius){
		this->bulletRadius = radius;
		assert(this->bulletRadius > 0);
	}

	void setBulletVel(float vel){
		this->buletVel = vel;
	}

	void setBulletCreator(bulletCreator *creator){
		this->creator = creator;
	}

	void setBulletData(bulletData &data){
		this->bullet = data;
	}


	//functions to be called during processing
	void setFacing(util::Angle facing){
		this->facing = facing;
	}



	void setBulletPos(vector2 pos){
		this->bulletPos = pos;
	}

	

	void Fire();
};



class objectMgr;

class gunProcessor : public objectProcessor{
private:
	void _fireShot(gunData *data, vector2 pos);
	objectMgr *objectManager;

public:
	gunProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager);
	

protected:
	void _Process(Object *obj, float dt);
	void _onObjectDeactivate(Object *obj);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("gunData");
	};
	
};
