#pragma once
#include "../../core/Object.h"
#include "../ObjProcessors/gunProcessor.h"
class viewProcess;
struct GunData;


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
	const Hash *enemyCollision;

	Object *parent;
	GunData gunData;

	float radius;

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

struct PickupFactoryInfo {};

Object *CreatePickup(PickupFactoryInfo &info);

//-----------------------------------------------------------

struct PlayerFactoryInfo {};

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
