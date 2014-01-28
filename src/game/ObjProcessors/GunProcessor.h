#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"

#include "BulletProcessor.h"
#include "StabProcessor.h"

#include "../../core/math/mathUtil.h"
#include "../../core/controlFlow/Cooldown.h"

struct GunData{
private:

	friend class GunProcessor;

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
	StabData stabData;

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

	void setBulletData(BulletData &bulletData){
		this->bulletData = bulletData;
	}

	void setStabData(StabData &stabData){
		this->stabData = stabData;
	}

};

class ObjectManager;

class GunProcessor : public ObjectProcessor{
private:
	void _fireShot(GunData *data, vector2 pos);
	ObjectManager *objectManager;
	viewProcess *viewProc;

public:
	GunProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager);
	

protected:
	void _Process(Object *obj, float dt);
	void _onObjectDeactivate(Object *obj);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("GunData");
	};
	
};
