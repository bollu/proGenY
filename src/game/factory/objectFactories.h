#pragma once
#include "../../core/componentSys/Object.h"
#include "../ObjProcessors/GunProcessor.h"
#include "../ObjProcessors/PickupProcessor.h"
#include "../ObjProcessors/CameraProcessor.h"

class viewProcess;


struct Terrain;

namespace ObjectFactories {

struct BoundaryFactoryInfo {
	viewProcess *viewProc;

	float thickness;
	vector2 levelDim;
};

Object *CreateBoundary(BoundaryFactoryInfo &info);


//-----------------------------------------------------------

struct GunFactoryInfo {
	viewProcess *viewProc;

	Object *parent;
	GunData gunData;
	vector2 pos;
};

Object *CreateGun(GunFactoryInfo &info);

//-----------------------------------------------------------
struct BulletFactoryInfo {
	viewProcess *viewProc;
	//const Hash *enemyCollision;

	BulletData bulletData;
	float radius;
	vector2 pos;

};

Object *CreateBullet(BulletFactoryInfo &info);

//-----------------------------------------------------------
struct EnemyFactoryInfo {
	viewProcess *viewProc;
	vector2 pos;

};

Object *CreateEnemy(EnemyFactoryInfo &info);
//-----------------------------------------------------------

struct PickupFactoryInfo {
	viewProcess *viewProc;

	PickupData pickup;
	float radius;
	vector2 pos;
};

Object *CreatePickup(PickupFactoryInfo &info);

//-----------------------------------------------------------

struct PlayerFactoryInfo {
	viewProcess *viewProc;
	CameraData cameraData;

	vector2 pos;
};

Object *CreatePlayer(PlayerFactoryInfo &info);

//-----------------------------------------------------------
struct TerrainFactoryInfo {

	TerrainFactoryInfo (Terrain &terrain_) : terrain(terrain_) {};
	
	viewProcess *viewProc;

	vector2 blockDim;
	Terrain &terrain;
};

Object *CreateTerrain(TerrainFactoryInfo &info);
};
