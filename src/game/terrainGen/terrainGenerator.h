#pragma once
#include <vector>
#include "../../core/math/vector.h"
#include "terrain.h"

struct phyData;
struct RenderData;

class viewProcess;

void genTerrain(Terrain &terrain, unsigned int seed);
//return player position in terrain coordinates (I hate coordinate systems :( 
vector2 getPlayerPosTerrain(Terrain &terrain, int terrainX);