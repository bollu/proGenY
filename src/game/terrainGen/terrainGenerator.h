#pragma once
#include <vector>
#include "../../core/math/vector.h"
#include "../../core/math/AABB.h"
#include "terrain.h"

struct PhyData;
struct RenderData;

class viewProcess;

void genTerrain(Terrain &terrain, unsigned int seed);
//return player position in terrain coordinates (I hate coordinate systems :( 
vector2 getPlayerPosTerrain(Terrain &terrain, int terrainX);


std::vector<AABB> genTerrainChunks(Terrain &terrain);