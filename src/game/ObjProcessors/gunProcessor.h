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
#include "../../util/Cooldown.h"

struct GunData{
private:

	friend class gunProcessor;

	int totalClipSize;
	//ticks down from currentAmmo to zero
	int currentClipSize;

	Cooldown<float> clipSize;
	Cooldown<float> clipCooldown;
	Cooldown<float> shotCooldown;

	util::Angle facing;
	vector2 bulletPos;
	float bulletRadius;
	float bulletVel;

	BulletData bulletData;

	void _Cooldown();
	void _Tick(float dt);
	bool _canFire();

public:
	GunData(){
		this->totalClipSize = this->currentClipSize = 0;	
	}
	
	void setClipSize(int totalClipSize){
		assert(totalClipSize > 0);
		this->totalClipSize = this->currentClipSize = totalClipSize;
	}

	void setClipCooldown(float totalClipCooldown){
		this->clipCooldown.setTotalTime(totalClipCooldown);
	}

	void setShotCooldown(float totalShotCooldown){
		this->shotCooldown.setTotalTime(totalShotCooldown);
	}

	
	void setBulletRadius(float radius){
		this->bulletRadius = radius;
		assert(this->bulletRadius > 0);
	}

	void setBulletVel(float vel){
		this->bulletVel = vel;
	}

	void setBulletData(BulletData &data){
		this->bulletData = data;
	}

};



class objectMgr;
class viewProcess;

class gunProcessor : public objectProcessor{
private:
	void _fireShot(GunData *data, vector2 pos);
	objectMgr *objectManager;
	viewProcess *viewProc;

public:
	gunProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager);
	
	void Process(float dt);
	void postProcess();
};